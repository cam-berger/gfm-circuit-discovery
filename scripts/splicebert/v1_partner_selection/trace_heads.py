#!/usr/bin/env python
"""Causally screen SpliceBERT attention heads on frozen cue-conflict examples.

This is a component-localization stage, not a complete circuit proof.  The
primary score is mean true-pair margin, where higher is better.  A head is only
promoted when its effect is directionally consistent across multiple ablation
nulls and a selected set beats size-matched random head sets.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

import torch

from gfmcd.adapters.splicebert import SpliceBertAdapter
from gfmcd.circuits.hf_bert import edit_heads, head_ablation_scan, matched_random_head_sets
from gfmcd.provenance import ExperimentManifest
from gfmcd.splicing.config import load_config, resolve_repo_path
from gfmcd.splicing.io import load_jsonl
from gfmcd.splicing.metrics import matched_random_pvalue
from gfmcd.splicing.readout import collate_pair_indices, true_pair_margin
from gfmcd.splicing.training import load_pair_checkpoint


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/splicebert_partner_selection/base.json")
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--examples",
        default="results/splicebert_partner_selection/baselines/val_local_conflicts.jsonl",
        help="frozen validation conflict set; test conflicts require a separately preregistered run",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--max-examples", type=int, default=32)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--random-sets", type=int, default=None)
    p.add_argument("--allow-secondary-regime", action="store_true")
    p.add_argument("--out-dir", default="results/splicebert_partner_selection/circuits/head_screen")
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[2]
    cfg = load_config(resolve_repo_path(repo, args.config))
    examples_path = resolve_repo_path(repo, args.examples)
    examples = load_jsonl(examples_path)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    if len(examples) < 2:
        raise RuntimeError("head screen requires at least two fixed examples")
    if any(x.split != "val" for x in examples):
        raise ValueError("discovery head screen must use validation examples only")

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
        raise SystemExit(
            f"refusing to use {regime!r} as the primary trace target; "
            "pass --allow-secondary-regime only for a labelled secondary analysis"
        )

    tokenized = adapter.tokenize([x.sequence for x in examples])
    pair_batch = collate_pair_indices(
        examples,
        token_offset=tokenized.prefix_tokens,
        device=adapter.device,
    )
    representation_layer = int(cfg["readout"]["representation_layer"])

    @torch.no_grad()
    def score_fn() -> float:
        outputs = adapter.forward(
            tokenized,
            output_hidden_states=True,
            output_attentions=False,
            grad=False,
        )
        hidden = outputs.hidden_states[representation_layer]
        logits = readout(hidden, pair_batch)
        return float(true_pair_margin(logits, pair_batch).mean())

    modes = tuple(cfg["circuit"].get("ablation_modes", ["zero", "mean", "resample"]))
    rows: list[dict] = []
    by_head: dict[tuple[int, int], list[float]] = {}
    for mode in modes:
        results = head_ablation_scan(adapter, score_fn, mode=mode)
        for result in results:
            row = result.to_dict()
            rows.append(row)
            by_head.setdefault((result.layer, result.head), []).append(result.effect)

    # Promote heads that are positive under at least two registered nulls.  Rank
    # by median effect so one pathological ablation does not dominate.
    ranked = []
    for head, effects in by_head.items():
        positive_modes = sum(effect > 0 for effect in effects)
        ranked.append(
            {
                "layer": head[0],
                "head": head[1],
                "median_effect": float(statistics.median(effects)),
                "min_effect": float(min(effects)),
                "positive_modes": positive_modes,
                "effects": effects,
            }
        )
    ranked.sort(key=lambda x: (x["positive_modes"] >= 2, x["median_effect"]), reverse=True)
    top_k = args.top_k or int(cfg["circuit"].get("default_component_set_size", 5))
    selected_rows = [x for x in ranked if x["positive_modes"] >= 2 and x["median_effect"] > 0][
        :top_k
    ]
    selected = [(x["layer"], x["head"]) for x in selected_rows]
    if not selected:
        raise RuntimeError("no head passed the registered cross-null promotion rule")

    baseline = score_fn()
    selected_effects = {}
    for mode in modes:
        with edit_heads(adapter, selected, mode=mode):
            intervened = score_fn()
        selected_effects[mode] = {
            "baseline_score": baseline,
            "intervened_score": intervened,
            "effect": baseline - intervened,
        }

    n_random = args.random_sets or int(cfg["circuit"].get("random_control_sets", 1000))
    random_sets = matched_random_head_sets(
        adapter,
        selected,
        n=n_random,
        seed=0,
        exclude_selected=True,
    )
    random_effects: list[float] = []
    # Mean ablation is the primary null for the set-level random comparison.
    for random_set in random_sets:
        with edit_heads(adapter, random_set, mode="mean"):
            random_effects.append(baseline - score_fn())
    observed = selected_effects["mean"]["effect"]
    null_report = {
        "mode": "mean",
        "observed_effect": observed,
        "random_effect_mean": float(statistics.mean(random_effects)),
        "random_effect_median": float(statistics.median(random_effects)),
        "p_value_add_one": matched_random_pvalue(observed, random_effects),
        "n_random_sets": len(random_effects),
    }

    run_name = checkpoint_path.parent.parent.name + "_" + checkpoint_path.parent.name
    out_dir = resolve_repo_path(repo, args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "head_effects.csv", rows)
    write_csv(
        out_dir / "head_consensus.csv",
        [{k: v for k, v in row.items() if k != "effects"} for row in ranked],
    )
    (out_dir / "random_set_effects.json").write_text(json.dumps(random_effects, indent=2) + "\n")
    report = {
        "regime": regime,
        "examples": str(examples_path),
        "n_examples": len(examples),
        "score": "mean true-pair logit margin; higher is better",
        "selected_heads": selected_rows,
        "selected_set_effects": selected_effects,
        "matched_random_null": null_report,
        "status": "COMPONENT_LOCALIZATION_ONLY",
        "next_required_steps": [
            "construct clean/corrupt aligned pairs with a non-zero behavioral gap",
            "activation-patch candidate heads and paths",
            "test necessity of the grouped circuit against matched random sets",
            "test sufficiency/mediation and replication across seeds",
        ],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    manifest = ExperimentManifest.create(
        f"{cfg['study_id']}:head_screen:{run_name}",
        sys.argv,
        repo=repo,
        config=cfg,
        notes=[
            "This run localizes candidate components; it does not establish a complete circuit.",
            "Validation conflict examples were frozen before tracing.",
        ],
    )
    manifest.add_input(examples_path)
    manifest.add_input(checkpoint_path)
    for path in out_dir.iterdir():
        if path.is_file():
            manifest.add_output(path)
    manifest.write(out_dir / "run_manifest.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
