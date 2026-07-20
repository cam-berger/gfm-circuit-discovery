#!/usr/bin/env python
"""Merge validated head-screen shards into a long-format matrix + per-head and concentration summaries.

Outputs (results/.../head_screen/):
  head_effect_matrix.jsonl    one row per (example, condition, layer, head)  [the exploratory matrix]
  head_effect_summary.json    per-(condition, layer, head) mean/CI + single-head-effect concentration

The concentration summary exists so the analysis does NOT assume mechanisms must be single-head sparse
(preserve C33): it reports the cumulative |single-head mean effect| captured by the top-k heads per
condition. IMPORTANT: this is a rank-CONCENTRATION statistic, NOT a faithfulness curve — heads interact
and single-head effects do not add. A true faithfulness-size curve requires JOINTLY patching ranked head
sets and measuring the resulting behavior (see the group-patch study). Do not read "heads_for_90pct" as
"90% of the mechanism needs this many heads".
"""
from __future__ import annotations

import argparse, json, math
from collections import defaultdict
from pathlib import Path

import numpy as np

CONDS = ("A_zero_ablation", "B", "C", "D")
METRICS = ("effect_margin", "effect_logprob", "effect_entropy", "effect_collateral")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    shards = sorted(out.glob("shard_*.json"))
    matrix_path = out / "head_effect_matrix.jsonl"
    per_head = defaultdict(lambda: defaultdict(list))  # (cond,l,h) -> metric -> [vals]
    n_ex = 0
    behav_path = out / "example_behavioral.jsonl"  # per-example baselines/counterfactuals the matrix drops
    with open(matrix_path, "w") as mf, open(behav_path, "w") as bf:
        for sp in shards:
            d = json.loads(sp.read_text())
            meta = d["example"]
            cds = d["conditions"]
            n_ex += 1
            # per-example behavioral table: A baseline + B/C/D baseline/counterfactual/behavioral_effect
            brow = {"example_id": meta["example_id"], "gene_id": meta["gene_id"], "site_type": meta["site_type"],
                    "chrom": meta.get("chrom"), "split": meta.get("split"),
                    "target_competitor_distance": meta.get("target_competitor_distance"),
                    "target_partner_distance": meta.get("target_partner_distance"),
                    "cluster_size": meta.get("cluster_size")}
            if "A_baseline" in cds:
                brow["A_baseline"] = cds["A_baseline"]
            for cond in ("B", "C", "D"):
                blk = cds.get(cond)
                if isinstance(blk, dict):
                    brow[f"{cond}_baseline"] = blk.get("baseline")
                    brow[f"{cond}_counterfactual"] = blk.get("counterfactual")
                    brow[f"{cond}_behavioral_effect"] = blk.get("behavioral_effect")
            bf.write(json.dumps(brow, sort_keys=True) + "\n")
            for cond in CONDS:
                block = d["conditions"][cond]
                recs = block if cond == "A_zero_ablation" else block["head_effects"]
                for r in recs:
                    l, h = r["layer"], r["head"]
                    row = {"example_id": meta["example_id"], "gene_id": meta["gene_id"],
                           "site_type": meta["site_type"], "cluster_size": meta["cluster_size"],
                           "target_competitor_distance": meta["target_competitor_distance"],
                           "condition": cond, "layer": l, "head": h}
                    if cond == "A_zero_ablation":
                        row["A_zero_effect_margin"] = r["A_zero_effect_margin"]
                        per_head[(cond, l, h)]["A_zero_effect_margin"].append(r["A_zero_effect_margin"])
                    else:
                        for m in METRICS:
                            row[m] = r[m]
                            per_head[(cond, l, h)][m].append(r[m])
                    mf.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {"n_examples": n_ex, "per_head": {}, "single_head_effect_concentration": {}}
    for (cond, l, h), md in per_head.items():
        key = f"{cond}|L{l}H{h}"
        summary["per_head"][key] = {m: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1) if len(v) > 1 else 0),
                                        "n": len(v)} for m, v in md.items()}
    # NOT faithfulness: cumulative |mean effect_margin| captured by top-k SINGLE heads per condition
    # (rank-concentration only; true faithfulness needs JOINT patching of ranked head sets).
    for cond in ("B", "C", "D"):
        means = sorted((abs(np.mean(per_head[(cond, l, h)]["effect_margin"]))
                        for l in range(6) for h in range(16) if (cond, l, h) in per_head), reverse=True)
        tot = sum(means) or 1.0
        cum = np.cumsum(means) / tot
        summary["single_head_effect_concentration"][cond] = {
            "top1_frac": float(cum[0]) if len(cum) else None,
            "top5_frac": float(cum[4]) if len(cum) > 4 else None,
            "top10_frac": float(cum[9]) if len(cum) > 9 else None,
            "heads_for_50pct_of_single_head_mass": int(np.searchsorted(cum, 0.5) + 1) if len(cum) else None,
            "heads_for_90pct_of_single_head_mass": int(np.searchsorted(cum, 0.9) + 1) if len(cum) else None,
            "note": "RANK-CONCENTRATION of single-head |mean margin effect| — NOT a faithfulness curve "
                    "(heads interact; single-head effects do not add). A shallow curve is consistent with a "
                    "distributed pattern (C33) but does not prove it; joint group-patching is required.",
        }
    (out / "head_effect_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"merged {len(shards)} shards, {n_ex} examples -> {matrix_path.name} + head_effect_summary.json + {behav_path.name}")
    for cond, g in summary["single_head_effect_concentration"].items():
        print(f"  {cond}: top1={g['top1_frac']:.2f} top5={g['top5_frac']} heads_for_90%={g['heads_for_90pct_of_single_head_mass']}")


if __name__ == "__main__":
    main()
