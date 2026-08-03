#!/usr/bin/env python
"""Paired per-head comparison: committed-operator screen vs motif-clean-alt screen (R44).

Both screens run the SAME 400-window discovery cohort with the SAME config; the only
difference is the destruction alt (``variants[0]`` = "CC" vs ``MOTIF_CLEAN_ALT`` = "AC").
Every head effect is therefore paired on (example, condition, layer, head).

Raw shards are 25 MB per screen and are NOT committed -- consistent with how the atlas has
always treated screen shards.  This aggregate is the compact artifact of record.

Emits `results/splicebert_head_context_atlas/head_screen_clean_alt_comparison_v1.json`.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest  # noqa: E402

CONDS = ("B", "C", "D")
METRIC = "effect_margin"


def load(dirpath: str) -> tuple[dict, dict]:
    """(per_head[(cond,l,h)] -> [effects], behavioural[cond] -> [margins]) keyed by example."""
    per_head: dict = defaultdict(dict)
    behav: dict = defaultdict(dict)
    for f in sorted(glob.glob(f"{dirpath}/shard_*.json")):
        s = json.load(open(f))
        eid = s["example"]["example_id"]
        for cond in CONDS:
            c = s["conditions"].get(cond)
            if not c:
                continue
            bm = c.get("behavioral_effect", {}).get("margin")
            if bm is not None:
                behav[cond][eid] = bm
            for he in c.get("head_effects", []):
                per_head[(cond, he["layer"], he["head"])][eid] = he[METRIC]
    return per_head, behav


def pearson(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = st.mean(x), st.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den if den else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--committed", default="results/splicebert_head_context_atlas/"
                                           "development/head_screen_cc_baseline")
    ap.add_argument("--clean", default="results/splicebert_head_context_atlas/head_screen_clean_alt")
    ap.add_argument("--out", default="results/splicebert_head_context_atlas/"
                                     "head_screen_clean_alt_comparison_v1.json")
    ap.add_argument("--overwrite-record", action="store_true")
    args = ap.parse_args()

    ph_a, bh_a = load(args.committed)
    ph_b, bh_b = load(args.clean)
    mf_a = json.load(open(Path(args.committed) / "run_manifest.json"))
    mf_b = json.load(open(Path(args.clean) / "run_manifest.json"))

    res: dict = {
        "operators": {"committed": mf_a.get("destroy_alt_mode"),
                      "clean": mf_b.get("destroy_alt_mode")},
        "planted_motif_counts": {"committed": mf_a.get("planted_motif_counts"),
                                 "clean": mf_b.get("planted_motif_counts")},
        "n_examples": {"committed": mf_a.get("n_examples"), "clean": mf_b.get("n_examples")},
        "metric": METRIC, "conditions": {},
    }

    for cond in CONDS:
        ea, eb = set(bh_a.get(cond, {})), set(bh_b.get(cond, {}))
        shared = sorted(ea & eb)
        entry: dict = {"n_examples_paired": len(shared),
                       "cohort_identical": ea == eb,
                       "behavioural": {}}
        if shared:
            xa = [bh_a[cond][e] for e in shared]
            xb = [bh_b[cond][e] for e in shared]
            entry["behavioural"] = {
                "committed_mean": st.mean(xa), "clean_mean": st.mean(xb),
                "delta": st.mean(xb) - st.mean(xa),
                "relative": (st.mean(xb) / st.mean(xa)) if st.mean(xa) else None,
                "pearson_r": pearson(xa, xb)}
        # per-head, averaged over the shared examples
        heads = sorted({k[1:] for k in ph_a if k[0] == cond} & {k[1:] for k in ph_b if k[0] == cond})
        ha, hb = [], []
        for (l, h) in heads:
            va = ph_a[(cond, l, h)]
            vb = ph_b[(cond, l, h)]
            common = [e for e in shared if e in va and e in vb]
            if not common:
                continue
            ha.append(st.mean([va[e] for e in common]))
            hb.append(st.mean([vb[e] for e in common]))
        order_a = sorted(range(len(heads)), key=lambda i: -ha[i])
        order_b = sorted(range(len(heads)), key=lambda i: -hb[i])
        top = lambda o, k: {heads[i] for i in o[:k]}  # noqa: E731
        entry["per_head"] = {
            "n_heads": len(heads),
            "pearson_r": pearson(ha, hb),
            "top1_committed": f"L{heads[order_a[0]][0]}H{heads[order_a[0]][1]}",
            "top1_clean": f"L{heads[order_b[0]][0]}H{heads[order_b[0]][1]}",
            "top1_same": heads[order_a[0]] == heads[order_b[0]],
            "top5_overlap": len(top(order_a, 5) & top(order_b, 5)),
            "top10_overlap": len(top(order_a, 10) & top(order_b, 10)),
            "top20_overlap": len(top(order_a, 20) & top(order_b, 20)),
            "top10_committed": [f"L{heads[i][0]}H{heads[i][1]}" for i in order_a[:10]],
            "top10_clean": [f"L{heads[i][0]}H{heads[i][1]}" for i in order_b[:10]],
            "max_abs_head_delta": max((abs(x - y) for x, y in zip(ha, hb)), default=None),
        }
        res["conditions"][cond] = entry

        b = entry["behavioural"]
        p = entry["per_head"]
        print(f"  {cond}: n={entry['n_examples_paired']} paired (cohort identical: "
              f"{entry['cohort_identical']})")
        if b:
            print(f"      behavioural margin  {b['committed_mean']:+.4f} -> {b['clean_mean']:+.4f}"
                  f"  ({b['delta']:+.4f}, rel {b['relative']:.3f}), per-example r={b['pearson_r']:.4f}")
        print(f"      per-head r={p['pearson_r']:.4f}  top1 {p['top1_committed']} -> "
              f"{p['top1_clean']} ({'SAME' if p['top1_same'] else 'CHANGED'})  "
              f"top5 {p['top5_overlap']}/5  top10 {p['top10_overlap']}/10  top20 {p['top20_overlap']}/20")

    p = guard_output_path(REPO / args.out, overwrite=args.overwrite_record)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1, default=float))
    ExperimentManifest.create(
        "splicebert_head_context_atlas:head_screen_clean_alt_comparison", sys.argv,
        config={"committed_dir": args.committed, "clean_dir": args.clean, "metric": METRIC,
                "note": "raw shards (25MB/screen) intentionally not committed"},
        status="VERIFICATION",
    ).write(p.with_suffix(".manifest.json"))
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
