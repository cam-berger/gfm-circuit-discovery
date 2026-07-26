#!/usr/bin/env python
"""ED1 matching pilot -- outcome-blind balance & yield for BOTH architecture axes.

Loads the cohort pilot JSON (annotation-derived covariates only; no model output exists
for the val split at this point) and reports, for each axis:

  * DOWNSTREAM axis (the classical intron- vs exon-definition transition), matched WITHIN
    the upstream flank band so a downstream contrast cannot smuggle in an upstream one.
  * UPSTREAM axis (the flank that is not a separation for either coupling -- the clean
    architecture manipulator), matched within the downstream band.

Yield + post-match SMDs here are what pin ``N`` in the preregistration.  Nothing about the
outcome is touched, so this may be run on the confirmatory split without unsealing it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.splicing.exon_intron import (  # noqa: E402
    EXACT_KEYS, axis_spec, covariate_row, match_pairs,
)


def smd(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


def evaluate(arch, treated, exact, label, axis):
    cov, cal = axis_spec(axis)
    pairs = match_pairs(arch, treated, exact=exact, covariates=cov, calipers=cal)
    mt = [arch[i] for i, _ in pairs]
    mc = [arch[j] for _, j in pairs]
    worst, table = 0.0, {}
    for k in cov:
        a = np.array([covariate_row(r)[k] for r in mc], float)
        b = np.array([covariate_row(r)[k] for r in mt], float)
        table[k] = smd(a, b)
        worst = max(worst, abs(table[k]))
    genes = {r["gene"] for r in mt + mc}
    print(f"\n=== {label} ===")
    print(f"  treated pool={sum(treated)}  control pool={len(treated)-sum(treated)}"
          f"  -> matched pairs={len(pairs)} ({len(pairs)/max(sum(treated),1):.1%})")
    print(f"  genes in matched cohort: {len(genes)}")
    for k, v in table.items():
        print(f"    {k:12s} SMD={v:+.3f}{'' if abs(v) < 0.1 else '   <-- IMBALANCED'}")
    print(f"  worst |SMD| = {worst:.3f} -> {'BALANCED' if worst < 0.1 else 'NOT BALANCED'}")
    both = sum(1 for r in mt + mc if r["n_sham_up"] > 0 and r["n_sham_dn"] > 0)
    print(f"  both-sham availability in matched cohort: {both/max(2*len(pairs),1):.3f}")
    return dict(label=label, n_pairs=len(pairs), n_genes=len(genes), worst_smd=worst,
                smd=table, both_sham_frac=both / max(2 * len(pairs), 1),
                pair_ids=[(arch[i]["example_id"], arch[j]["example_id"]) for i, j in pairs])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="results/splicebert_coupling/development/ed1_cohort_pilot_val.json")
    # Default is COMPUTED from the cohort split, never a literal record path: a new
    # runner must not default --out to an artifact in results/MANIFEST.csv (the
    # unguarded_out_defaults inventory is for legacy runners and must not grow).
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite-record", action="store_true",
                    help="replace an existing artifact of record (results are immutable by default)")
    args = ap.parse_args()

    data = json.loads((REPO / args.cohort).read_text())
    arch = [dict(r, up_band=r["cell"][0], dn_band=r["cell"][1])
            for r in data["rows"] if r["cell"] is not None]
    print(f"cohort={args.cohort}  split={data['split']}  in-2x2 units={len(arch)}")
    for ax in ("down", "up"):
        print(f"  axis {ax:5s}: covariates={axis_spec(ax)[0]}  calipers={axis_spec(ax)[1]}")

    # Treat the SMALLER arm as "treated": greedy matching can yield at most
    # min(|arm|) pairs, and processing the smaller arm spends the good matches on it.
    up_l = sum(r["up_band"] == "L" for r in arch)
    res = [
        evaluate(arch, [r["dn_band"] == "L" for r in arch], EXACT_KEYS["down"],
                 "DOWNSTREAM axis (down-LONG vs down-SHORT), exact on upstream band", "down"),
        evaluate(arch, [r["up_band"] == ("L" if up_l < len(arch) - up_l else "S")
                        for r in arch], EXACT_KEYS["up"],
                 "UPSTREAM axis (up-LONG vs up-SHORT), exact on downstream band", "up"),
    ]

    # per-cell yield on the primary axis, for the 2x2 report
    print("\n=== primary-axis matched pairs by upstream band ===")
    idx = {r["example_id"]: r for r in arch}
    by: dict[str, list] = {}
    for t, c in res[0]["pair_ids"]:
        by.setdefault(idx[t]["up_band"], []).append((t, c))
    for k in sorted(by):
        ng = len({idx[t]["gene"] for t, _ in by[k]} | {idx[c]["gene"] for _, c in by[k]})
        print(f"  up-{k}:  {len(by[k])} pairs, {ng} genes   (cell {k}L vs cell {k}S)")
    used_t = [t for t, _ in res[0]["pair_ids"]]
    used_c = [c for _, c in res[0]["pair_ids"]]
    assert len(set(used_t)) == len(used_t), "a treated unit was matched twice"
    assert len(set(used_c)) == len(used_c), "a control unit was reused"

    out = guard_output_path(
        REPO / (args.out or f"results/splicebert_coupling/development/"
                            f"ed1_matching_pilot_{data['split']}.json"),
        overwrite=args.overwrite_record)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(cohort=args.cohort, split=data["split"],
                                   calipers={a: axis_spec(a)[1] for a in ("down","up")}, results=res), indent=1))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
