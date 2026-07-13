#!/usr/bin/env python
"""Zero-shot partner-dependency pilot using SpliceBERT's native MLM head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

from gfmcd.adapters.splicebert import SpliceBertAdapter
from gfmcd.provenance import ExperimentManifest
from gfmcd.splicing.config import load_config, resolve_repo_path
from gfmcd.splicing.io import load_jsonl
from gfmcd.splicing.metrics import grouped_bootstrap
from gfmcd.splicing.zero_shot import pair_native_dependency


def choose_pairs(example, max_pairs, baseline=None):
    indexed = list(enumerate(example.pairs))
    true = example.true_pair_index
    if baseline is not None:
        scores = baseline.score_example(example)
        indexed.sort(key=lambda x: float(scores[x[0]]), reverse=True)
    else:
        tp = example.true_pair
        indexed.sort(
            key=lambda x: (
                abs(x[1].acceptor_pos - tp.acceptor_pos)
                + abs(x[1].donor_pos - tp.donor_pos),
                abs(x[1].exon_length - tp.exon_length),
            )
        )
    kept = indexed[:max_pairs]
    if true not in {i for i, _ in kept}:
        kept[-1] = (true, example.true_pair)
    return sorted(kept, key=lambda x: x[0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/splicebert_partner_selection/base.json")
    p.add_argument(
        "--examples",
        default="results/splicebert_partner_selection/baselines/val_local_conflicts.jsonl",
    )
    p.add_argument("--local-baseline", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--max-examples", type=int, default=50)
    p.add_argument("--max-pairs", type=int, default=8)
    p.add_argument("--partner-flank", type=int, default=2)
    p.add_argument("--out-dir", default="results/splicebert_partner_selection/behavior/native_dependency")
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[2]
    cfg = load_config(resolve_repo_path(repo, args.config))
    examples_path = resolve_repo_path(repo, args.examples)
    examples = load_jsonl(examples_path)[: args.max_examples]
    if not examples or any(x.split != "val" for x in examples):
        raise ValueError("native dependency pilot requires non-empty validation examples")

    baseline = None
    baseline_path = None
    if args.local_baseline:
        baseline_path = resolve_repo_path(repo, args.local_baseline)
        with baseline_path.open("rb") as f:
            baseline = pickle.load(f)

    model_cfg = cfg["model"]
    adapter = SpliceBertAdapter.from_pretrained(
        resolve_repo_path(repo, model_cfg["path"]),
        tokenizer_mode=model_cfg.get("tokenizer_mode", "auto"),
        device=args.device,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        local_files_only=bool(model_cfg.get("local_files_only", True)),
        model_class="masked_lm",
    )

    rows = []
    margins = []
    correct = []
    for n, example in enumerate(examples, start=1):
        candidates = choose_pairs(example, args.max_pairs, baseline)
        pair_rows = []
        for original_index, pair in candidates:
            dep = pair_native_dependency(
                adapter,
                example,
                pair,
                partner_flank=args.partner_flank,
            )
            pair_rows.append(
                {
                    "pair_index": original_index,
                    "acceptor_pos": pair.acceptor_pos,
                    "donor_pos": pair.donor_pos,
                    **dep.to_dict(),
                }
            )
        scores = np.asarray([x["symmetric_dependency"] for x in pair_rows])
        original_indices = [x["pair_index"] for x in pair_rows]
        true_local = original_indices.index(example.true_pair_index)
        decoys = np.delete(scores, true_local)
        margin = float(scores[true_local] - decoys.max())
        prediction = original_indices[int(scores.argmax())]
        margins.append(margin)
        correct.append(prediction == example.true_pair_index)
        rows.append(
            {
                "example_id": example.example_id,
                "gene_id": example.gene_id,
                "true_pair_index": example.true_pair_index,
                "predicted_pair_index": prediction,
                "true_margin": margin,
                "candidate_subset_size": len(pair_rows),
                "pairs": pair_rows,
            }
        )
        print(f"[{n}/{len(examples)}] {example.example_id} margin={margin:.4f}")

    report = {
        "n": len(rows),
        "candidate_subset_max": args.max_pairs,
        "partner_flank": args.partner_flank,
        "top1_accuracy_within_candidate_subset": float(np.mean(correct)),
        "positive_margin_rate": float(np.mean(np.asarray(margins) > 0)),
        "margin_by_gene": grouped_bootstrap(
            margins, [x.gene_id for x in examples], seed=0
        ),
        "claim_scope": [
            "This is a native masked-LM dependency assay with no trained pair readout.",
            "It is sensitive to N-replacement and candidate subsampling; report both explicitly.",
            "A positive result motivates circuit tracing but does not localize or validate a circuit.",
        ],
    }
    out_dir = resolve_repo_path(repo, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / "per_example.jsonl").write_text(
        "".join(json.dumps(x, sort_keys=True) + "\n" for x in rows)
    )

    manifest = ExperimentManifest.create(
        f"{cfg['study_id']}:native_dependency",
        sys.argv,
        repo=repo,
        config=cfg,
        notes=[
            "Validation only; test split remained unopened.",
            "Candidate subsampling always retains the annotated true pair.",
        ],
    )
    manifest.add_input(examples_path)
    if baseline_path:
        manifest.add_input(baseline_path)
    manifest.add_output(out_dir / "report.json")
    manifest.add_output(out_dir / "per_example.jsonl")
    manifest.write(out_dir / "run_manifest.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
