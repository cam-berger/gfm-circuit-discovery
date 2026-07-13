#!/usr/bin/env python
"""Fit the registered local PWM+length baseline and freeze a conflict set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

from gfmcd.provenance import ExperimentManifest
from gfmcd.splicing.baselines import LocalPairBaseline, baseline_conflict_examples
from gfmcd.splicing.config import load_config, resolve_repo_path
from gfmcd.splicing.io import load_jsonl, save_jsonl


def metrics(baseline: LocalPairBaseline, examples):
    pred = np.asarray([baseline.predict(x) for x in examples])
    target = np.asarray([x.true_pair_index for x in examples])
    margins = np.asarray([baseline.true_margin(x) for x in examples])
    return {
        "n": len(examples),
        "top1_accuracy": float((pred == target).mean()) if len(examples) else None,
        "mean_true_margin": float(margins.mean()) if len(examples) else None,
        "positive_margin_rate": float((margins > 0).mean()) if len(examples) else None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/splicebert_partner_selection/base.json")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", default="results/splicebert_partner_selection/baselines")
    p.add_argument("--include-test", action="store_true", help="open the held-out test split only after preregistration")
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[2]
    cfg = load_config(resolve_repo_path(repo, args.config))
    data_dir = resolve_repo_path(repo, args.data_dir or cfg["data"]["output_dir"])
    out_dir = resolve_repo_path(repo, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = load_jsonl(data_dir / "train.jsonl")
    val = load_jsonl(data_dir / "val.jsonl")
    test = load_jsonl(data_dir / "test.jsonl") if args.include_test else []

    baseline = LocalPairBaseline.fit(train)
    with (out_dir / "local_pair_baseline.pkl").open("wb") as f:
        pickle.dump(baseline, f)
    report = {
        "baseline": "acceptor PWM + donor PWM + empirical exon-length prior",
        "train": metrics(baseline, train),
        "val": metrics(baseline, val),
        "test": metrics(baseline, test) if args.include_test else {"status": "HELD_OUT"},
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    val_conflicts = baseline_conflict_examples(baseline, val)
    test_conflicts = baseline_conflict_examples(baseline, test) if args.include_test else []
    save_jsonl(val_conflicts, out_dir / "val_local_conflicts.jsonl")
    if args.include_test:
        save_jsonl(test_conflicts, out_dir / "test_local_conflicts.jsonl")

    manifest = ExperimentManifest.create(
        f"{cfg['study_id']}:local_baseline",
        sys.argv,
        repo=repo,
        config=cfg,
        notes=[
            "The conflict set contains natural examples where the frozen local baseline selects a decoy.",
            "Do not alter the baseline after inspecting circuit results.",
        ],
    )
    for name in (("train", "val", "test") if args.include_test else ("train", "val")):
        manifest.add_input(data_dir / f"{name}.jsonl")
    for path in out_dir.iterdir():
        if path.is_file():
            manifest.add_output(path)
    manifest.write(out_dir / "run_manifest.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
