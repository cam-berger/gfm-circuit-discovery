#!/usr/bin/env python
"""Build the frozen tracing splits BEFORE any component intervention (plan G3/G4).

- D  (discovery):    val conflicts on which heads are SELECTED.
- Vc (confirmation): val conflicts, GENE-DISJOINT from D, on which every
                     pre-registered pass rule (necessity / mediation) is SCORED.
- O  (ordinary):     val NON-conflict exons, distribution-matched to Vc on
                     candidate count + exon length, for the specificity test
                     (circuit effect on conflicts must exceed effect on O).

Gene-disjoint D/Vc prevents head selection from leaking into the confirmatory
numbers through gene-specific features. Everything is deterministic (fixed seed).
The test split is never touched here.
"""
from __future__ import annotations

import argparse
import bisect
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics as st
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gfmcd.splicing.io import load_jsonl, save_jsonl  # noqa: E402
from gfmcd.provenance import sha256_file  # noqa: E402


def exon_bin(length: int, edges: list[int]) -> int:
    return bisect.bisect_right(edges, length)


def stratified_sample(examples, edges, target, rng):
    """Sample `target` examples keeping the exon-length-bin distribution, spread across genes."""
    by_bin = defaultdict(list)
    for x in examples:
        by_bin[exon_bin(x.true_pair.exon_length, edges)].append(x)
    total = len(examples)
    picked = []
    for b, xs in by_bin.items():
        rng.shuffle(xs)
        want = round(target * len(xs) / total)
        picked.extend(xs[:want])
    rng.shuffle(picked)
    # exact size: trim, or top up from the remainder if rounding undershot
    if len(picked) > target:
        picked = picked[:target]
    elif len(picked) < target:
        chosen = {x.example_id for x in picked}
        rest = [x for x in examples if x.example_id not in chosen]
        rng.shuffle(rest)
        picked.extend(rest[: target - len(picked)])
    return picked


def match_ordinary(vc, ordinary, rng):
    """Greedy nearest match: exact candidate count, then nearest exon length; no reuse."""
    by_cc = defaultdict(list)
    for x in ordinary:
        by_cc[len(x.pairs)].append(x)
    for cc in by_cc:
        by_cc[cc].sort(key=lambda x: x.true_pair.exon_length)
    ccs = sorted(by_cc)
    used = set()
    matched = []
    for v in vc:
        vcc, vel = len(v.pairs), v.true_pair.exon_length
        # search candidate-count buckets outward from vcc
        order = sorted(ccs, key=lambda c: abs(c - vcc))
        best = None
        for c in order[:5]:
            lst = by_cc[c]
            els = [x.true_pair.exon_length for x in lst]
            i = bisect.bisect_left(els, vel)
            for j in range(max(0, i - 3), min(len(lst), i + 4)):
                cand = lst[j]
                if cand.example_id in used:
                    continue
                cost = (abs(len(cand.pairs) - vcc), abs(cand.true_pair.exon_length - vel),
                        0 if cand.gene_id == v.gene_id else 1)
                if best is None or cost < best[0]:
                    best = (cost, cand)
            if best is not None and best[0][0] == 0 and best[0][1] <= 5:
                break
        if best is None:  # exhausted; take any unused
            pool = [x for x in ordinary if x.example_id not in used]
            best = ((0, 0, 1), rng.choice(pool))
        used.add(best[1].example_id)
        matched.append(best[1])
    return matched


def dist(xs):
    el = [x.true_pair.exon_length for x in xs]
    cc = [len(x.pairs) for x in xs]
    return {"n": len(xs), "n_genes": len({x.gene_id for x in xs}),
            "exon_len_median": int(st.median(el)), "exon_len_mean": round(st.mean(el), 1),
            "cand_count_median": int(st.median(cc))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--conflicts", default="colab_results/splicebert_run/baselines/val_local_conflicts.jsonl")
    p.add_argument("--val", default="data/processed/splicebert_partner_selection/v1/val.jsonl")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="results/splicebert_partner_selection/circuits/discovery")
    args = p.parse_args()

    rng = random.Random(args.seed)
    conflicts = load_jsonl(REPO / args.conflicts)
    if any(x.split != "val" for x in conflicts):
        raise SystemExit("conflict set must be validation-only")
    conflict_ids = {x.example_id for x in conflicts}

    # exon-length bin edges = quintiles of the conflict pool (shared by D and Vc)
    el_sorted = sorted(x.true_pair.exon_length for x in conflicts)
    edges = [el_sorted[int(q * len(el_sorted))] for q in (0.2, 0.4, 0.6, 0.8)]

    # gene-disjoint partition of conflict genes
    genes = sorted({x.gene_id for x in conflicts})
    rng.shuffle(genes)
    half = len(genes) // 2
    d_genes, vc_genes = set(genes[:half]), set(genes[half:])
    d_pool = [x for x in conflicts if x.gene_id in d_genes]
    vc_pool = [x for x in conflicts if x.gene_id in vc_genes]

    D = stratified_sample(d_pool, edges, args.size, rng)
    Vc = stratified_sample(vc_pool, edges, args.size, rng)

    # ordinary pool = validation exons that are NOT conflicts
    val = load_jsonl(REPO / args.val)
    ordinary_pool = [x for x in val if x.split == "val" and x.example_id not in conflict_ids]
    O = match_ordinary(Vc, ordinary_pool, rng)

    # --- invariants (fail loudly rather than silently produce a bad split) ---
    d_ids, vc_ids, o_ids = ({x.example_id for x in s} for s in (D, Vc, O))
    assert len(d_ids) == len(vc_ids) == args.size, "D/Vc not exact size"
    assert not (d_ids & vc_ids), "D and Vc share examples"
    assert not (d_genes & vc_genes), "D and Vc share genes"
    assert not (o_ids & conflict_ids), "O contains conflicts"
    assert len(o_ids) == len(O), "O has duplicate matches"

    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    save_jsonl(D, out / "discovery.jsonl")
    save_jsonl(Vc, out / "confirmation.jsonl")
    save_jsonl(O, out / "ordinary.jsonl")
    manifest = {
        "seed": args.seed, "size": args.size,
        "exon_length_bin_edges": edges,
        "source_conflicts": args.conflicts, "source_val": args.val,
        "gene_disjoint_D_Vc": True,
        "distributions": {"D": dist(D), "Vc": dist(Vc), "O": dist(O)},
        "hashes": {
            "discovery.jsonl": sha256_file(out / "discovery.jsonl"),
            "confirmation.jsonl": sha256_file(out / "confirmation.jsonl"),
            "ordinary.jsonl": sha256_file(out / "ordinary.jsonl"),
        },
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
