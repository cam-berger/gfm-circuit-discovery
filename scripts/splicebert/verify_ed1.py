#!/usr/bin/env python
"""Independent re-derivation of the ED1 primary — deliberately NOT sharing code with the runner.

Adversarial verification needs a second path, not a second call.  This script:

  1. recomputes `C_exon` / `C_intron` straight from the raw per-state margins in the score
     artifact, with its own arithmetic;
  2. rebuilds the matched cohort with its own simple greedy matcher;
  3. computes the standard errors ANALYTICALLY with the Cameron-Gelbach-Miller two-way
     cluster-robust formula `V_a + V_b − V_ab` — no bootstrap, no multipliers, so it shares
     no machinery with the runner's inference and cannot inherit its bugs (the runner's
     first inference scheme over-rejected at FPR ~0.19; see redteam B4);
  4. reports the discrepancy against the runner's published numbers.

Any material disagreement means one of the two paths is wrong, and both get audited.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def two_way_cluster_se(dev, ga, gb):
    """CGM two-way cluster-robust SE of a mean: V_a + V_b − V_ab."""
    n = len(dev)

    def one_way(keys):
        s: dict = {}
        for k, d in zip(keys, dev):
            s[k] = s.get(k, 0.0) + d
        return sum(v * v for v in s.values()) / (n * n)

    v_a = one_way(list(ga))
    v_b = one_way(list(gb))
    v_ab = one_way([f"{a}|{b}" for a, b in zip(ga, gb)])
    return math.sqrt(max(v_a + v_b - v_ab, 0.0))


def greedy_match(rows, treated, cov, cal, exact):
    """A second, independently written 1:1 matcher (same frozen spec, different code)."""
    Z = np.array([[float(r[c]) for c in cov] for r in rows])
    sd = Z.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    Zs = (Z - Z.mean(axis=0)) / sd
    t = [i for i in range(len(rows)) if treated[i]]
    c = [i for i in range(len(rows)) if not treated[i]]
    t.sort(key=lambda i: rows[i]["example_id"])
    taken, out = set(), []
    for i in t:
        best, bd = None, np.inf
        for j in c:
            if j in taken:
                continue
            if any(rows[i][k] != rows[j][k] for k in exact):
                continue
            if any(abs(rows[i][k] - rows[j][k]) > w for k, w in cal.items()):
                continue
            d = float(np.linalg.norm(Zs[i] - Zs[j]))
            if d < bd:
                best, bd = j, d
        if best is not None:
            taken.add(best)
            out.append((i, best))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/splicebert_coupling/development/ed1_scores_val.json")
    ap.add_argument("--result", default="results/splicebert_coupling/ed1_architecture_boundary_pairing.json")
    args = ap.parse_args()

    rows = json.loads((REPO / args.scores).read_text())["rows"]
    pub = json.loads((REPO / args.result).read_text())

    # 1. estimands, recomputed from raw state margins (index 0 = linear observable)
    for r in rows:
        f = {k: v[0] for k, v in r["f"].items()}
        r["C_exon"] = f["N"] - f["U-"]
        r["C_intron"] = f["N"] - f["D-"]
        r["log_L_up"] = math.log1p(r["L_up"] or 0)
        r["log_iso_don"] = math.log1p(r["n_iso_don"])
        r["log_iso_acc"] = math.log1p(r["n_iso_acc"])
        r["up_band"] = r["cell"][0] if r["cell"] else None
        r["dn_band"] = r["cell"][1] if r["cell"] else None

    arch = [r for r in rows if r["cell"]]
    cov = ["L_exon", "gc_exon", "gc_intron", "s_acc", "s_next", "s_don", "log_iso_don",
           "log_iso_acc", "donor_pos", "acc_pos", "log_L_up"]
    cal = {"L_exon": 10.0, "gc_intron": 0.03, "gc_exon": 0.05, "log_L_up": 0.5}
    pairs = greedy_match(arch, [r["dn_band"] == "L" for r in arch], cov, cal,
                         ("up_band", "gene_type"))

    ga = [arch[i]["gene"] for i, _ in pairs]
    gb = [arch[j]["gene"] for _, j in pairs]
    e = np.array([arch[i]["C_exon"] - arch[j]["C_exon"] for i, j in pairs])
    ii = np.array([arch[i]["C_intron"] - arch[j]["C_intron"] for i, j in pairs])
    dd = e - ii

    print("=" * 78)
    print(f"INDEPENDENT RE-DERIVATION   pairs={len(pairs)}  "
          f"genes={len(set(ga) | set(gb))}   (runner: {pub['cohort']['n_pairs']} pairs)")
    print("  analytic two-way cluster-robust SE (Cameron-Gelbach-Miller), no bootstrap")
    print("-" * 78)
    rep = pub["primary"]
    for name, v, key in (("E_switch", e, "E_switch"), ("I_switch", ii, "I_switch"),
                         ("dd", dd, "dd")):
        m = float(v.mean())
        se = two_way_cluster_se(v - m, ga, gb)
        lo, hi = m - 1.96 * se, m + 1.96 * se
        r = rep[key]
        print(f"  {name:9s} mine={m:+.4f} [{lo:+.4f},{hi:+.4f}] (se={se:.4f})   "
              f"runner={r['mean']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}]   "
              f"Δmean={m - r['mean']:+.5f}")
        agree = (not (lo <= 0 <= hi)) == (not (r["lo"] <= 0 <= r["hi"]))
        print(f"            same reject/accept decision: {agree}")
    print("=" * 78)


if __name__ == "__main__":
    main()
