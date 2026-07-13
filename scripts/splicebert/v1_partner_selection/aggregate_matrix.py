#!/usr/bin/env python
"""Regenerate the validation-matrix summary from the per-run artifacts.

Reproducible replacement for the earlier one-off aggregation (which went stale after
the additive-random-frozen control was re-run -- RRR finding 5).

Metric note (RRR finding 4): `val_top1` (top-1 accuracy) and `val_pmr`
(positive-margin rate) are the **same statistic up to rank-1 ties** -- the score is the
true-vs-best-decoy margin, so margin > 0 <=> the true pair outranks every decoy. They
differ only when the true pair exactly ties the best decoy (argmax tie-breaks, strict
margin>0 does not); that is measure-zero for continuous logits and ~1-2/4000 for the
degenerate random-frozen control. They are NOT two independent corroborating metrics.
`positive_margin_rate` is the registered gate metric; `top1_accuracy` is retained only
as the conventional name. The identity (and its tie caveat) is asserted in
tests/test_metric_identity.py.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="colab_results/splicebert_run/runs")
    p.add_argument("--out", default="colab_results/summary.csv")
    args = p.parse_args()
    runs = REPO / args.runs
    rows = []
    for d in sorted(p.name for p in runs.iterdir() if p.is_dir()):
        regime, seed = d.rsplit("_seed", 1)
        tm = runs / d / "train_metrics.json"
        bc = runs / d / "behavior_conflict.json"
        if not tm.exists():
            rows.append({"regime": regime, "seed": seed, "status": "INCOMPLETE"})
            continue
        vm = json.loads(tm.read_text()).get("val_metrics", {})
        conf = ""
        if bc.exists():
            conf = json.loads(bc.read_text()).get("conditions", {}).get("clean", {}).get(
                "metrics", {}).get("positive_margin_rate", "")
        rows.append({
            "regime": regime, "seed": seed, "status": "ok",
            "val_top1": round(vm.get("top1_accuracy", 0), 4),
            "val_pmr": round(vm.get("positive_margin_rate", 0), 4),
            "val_mean_margin": round(vm.get("mean_margin", 0), 3),
            "conflict_pmr": round(conf, 4) if conf != "" else "",
        })
    fields = ["regime", "seed", "status", "val_top1", "val_pmr", "val_mean_margin", "conflict_pmr"]
    out = REPO / args.out
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    n_ok = sum(r["status"] == "ok" for r in rows)
    print(f"wrote {out}  ({n_ok}/{len(rows)} runs complete)")
    for r in rows:
        if r["status"] != "ok":
            print("  INCOMPLETE:", r["regime"], "seed", r["seed"])


if __name__ == "__main__":
    main()
