"""Audit: is the partner-selection task solved by pure SITE DETECTION + length prior?

Pure data + frozen baseline pickle. No model needed.
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/Users/cberg/Research/projects/test/GFM_SpliceBERT_Circuit_Discovery_Implementation/gfm_splicebert_implementation")
sys.path.insert(0, str(ROOT / "src"))

from gfmcd.splicing.schema import SplicePairExample  # noqa: E402

BASE = ROOT / "colab_results/splicebert_run/baselines"

with open(BASE / "local_pair_baseline.pkl", "rb") as fh:
    baseline = pickle.load(fh)

examples = [
    SplicePairExample.from_json(line)
    for line in (BASE / "val_local_conflicts.jsonl").read_text().splitlines()
    if line.strip()
]
print(f"loaded {len(examples)} conflict examples")
print(f"length_weight={baseline.length_weight} "
      f"length_prior range=[{baseline.length_prior.min_length},{baseline.length_prior.max_length}] "
      f"log_prob min={baseline.length_prior.log_prob.min():.3f} max={baseline.length_prior.log_prob.max():.3f}")

# ---------------------------------------------------------------- 1. ORACLE
def oracle_scores(ex: SplicePairExample) -> np.ndarray:
    tp = ex.true_pair
    ta, td = tp.acceptor_pos, tp.donor_pos
    lw, lp = baseline.length_weight, baseline.length_prior
    return np.asarray(
        [
            float(p.acceptor_pos == ta)
            + float(p.donor_pos == td)
            + lw * lp.score(p.exon_length)
            for p in ex.pairs
        ],
        dtype=np.float64,
    )


# oracle WITHOUT the length prior (pure site detection, ties broken pessimistically)
def oracle_scores_nolen(ex: SplicePairExample) -> np.ndarray:
    tp = ex.true_pair
    return np.asarray(
        [float(p.acceptor_pos == tp.acceptor_pos) + float(p.donor_pos == tp.donor_pos)
         for p in ex.pairs],
        dtype=np.float64,
    )


def pmr_and_top1(scorer):
    pmr = top1 = 0
    margins = []
    for ex in examples:
        s = scorer(ex)
        t = s[ex.true_pair_index]
        d = np.delete(s, ex.true_pair_index)
        best_d = d.max()
        margins.append(t - best_d)
        if t > best_d:
            pmr += 1
        # top-1: strict argmax uniqueness required
        if int(np.argmax(s)) == ex.true_pair_index and t > best_d:
            top1 += 1
    return pmr / len(examples), top1 / len(examples), np.asarray(margins)


for name, fn in [("ORACLE site-detector + length prior", oracle_scores),
                 ("ORACLE site-detector, NO length prior", oracle_scores_nolen)]:
    pmr, top1, m = pmr_and_top1(fn)
    print(f"\n[1] {name}")
    print(f"    PMR (true > best decoy, strict) = {pmr:.4f}")
    print(f"    top-1 accuracy                  = {top1:.4f}")
    print(f"    margin: median={np.median(m):.3f} min={m.min():.3f} "
          f"frac<=0={np.mean(m <= 0):.4f}")

# how do the oracle's failures look?
fails = []
for ex in examples:
    s = oracle_scores(ex)
    t = s[ex.true_pair_index]
    d = np.delete(s, ex.true_pair_index)
    if t <= d.max():
        fails.append(ex)
print(f"    oracle(+len) failures: {len(fails)}")

# ------------------------------------------------ 2. structure of best decoy
bd = Counter()
for ex in examples:
    s = baseline.score_example(ex)
    s[ex.true_pair_index] = -np.inf
    j = int(np.argmax(s))
    p, tp = ex.pairs[j], ex.true_pair
    sa = p.acceptor_pos == tp.acceptor_pos
    sd = p.donor_pos == tp.donor_pos
    bd["shares TRUE acceptor only" if sa and not sd else
       "shares TRUE donor only" if sd and not sa else
       "shares BOTH (impossible)" if sa and sd else
       "shares NEITHER"] += 1
print("\n[2] best decoy (by PWM baseline) composition:")
for k, v in bd.most_common():
    print(f"    {k:28s} {v:6d}  {v/len(examples):.4f}")

# ------------------------------------------------------- 3. candidate counts
npairs = np.array([len(ex.pairs) for ex in examples])
nacc = np.array([len(ex.acceptors) for ex in examples])
ndon = np.array([len(ex.donors) for ex in examples])
print("\n[3] candidate counts (per example):")
for nm, a in [("pairs", npairs), ("acceptors", nacc), ("donors", ndon)]:
    print(f"    {nm:10s} mean={a.mean():7.2f} median={np.median(a):6.1f} "
          f"min={a.min()} max={a.max()}  frac at cap(512)={np.mean(a==512):.3f}"
          if nm == "pairs" else
          f"    {nm:10s} mean={a.mean():7.2f} median={np.median(a):6.1f} "
          f"min={a.min()} max={a.max()}")

# ---------------------------------------- 4. ambiguity / real-site annotation
rec = json.loads((BASE / "val_local_conflicts.jsonl").read_text().splitlines()[0])
print("\n[4] record keys:", sorted(rec.keys()))
print("    metadata keys:", sorted(rec["metadata"].keys()))
print("    acceptor candidate keys:", sorted(rec["acceptors"][0].keys()))
print("    sample acceptor entries:", rec["acceptors"][:3])
# does ANY candidate carry a label marking it as a real annotated site?
labelled = set()
for ex in examples[:500]:
    for c in list(ex.acceptors) + list(ex.donors):
        labelled.add(tuple(sorted(vars(c).keys())) if hasattr(c, "__dict__") else None)
print("    site candidate fields:", labelled)
loc = Counter()
for ex in examples:
    loc[ex.acceptors[0].local_score is None] += 1
print("    local_score is None on first acceptor:", dict(loc))

# how many candidate acceptors/donors carry the canonical AG / GT motif?
mot_a, mot_d = Counter(), Counter()
for ex in examples:
    for c in ex.acceptors:
        mot_a[c.motif] += 1
    for c in ex.donors:
        mot_d[c.motif] += 1
print("    acceptor candidate motifs:", mot_a.most_common())
print("    donor candidate motifs:", mot_d.most_common())
tm_a = Counter(ex.metadata.get("true_acceptor_motif") for ex in examples)
tm_d = Counter(ex.metadata.get("true_donor_motif") for ex in examples)
print("    TRUE acceptor motif:", tm_a.most_common())
print("    TRUE donor motif:", tm_d.most_common())
