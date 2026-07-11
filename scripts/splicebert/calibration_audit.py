#!/usr/bin/env python
"""Calibration audit — the statistical exit criterion for the tracing null machinery.

Runs PURE NOISE through the same selection + p-value + BH-FDR path the head screen
uses (max over a search of head effects, calibrated against a same-search null), and
checks that on noise it finds nothing:

  - per-search MAXIMUM p-values are Uniform(0,1)  (KS p >= 0.01)   [only maxima are uniform]
  - false-positive rate at alpha across all scores <= alpha (small tolerance)
  - ~0 discoveries at the operational FDR threshold

Search sizes mirror the real screen: the 16-head shortlist and the full 96 heads
(6 layers x 16). Exits non-zero if any check fails, so it can gate CI / a run.

This audits the STATISTICS. The MODEL-level negative control is separate and
complementary: the additive-random-frozen backbone (no pretrained circuit) run
through the real head screen must also come up empty — run that on GPU alongside
the tracing seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gfmcd.rigor import audit_calibration, required_null_draws  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--search-sizes", type=int, nargs="+", default=[16, 48, 96])
    p.add_argument("--n-searches", type=int, default=200,
                   help="number of noise searches; the KS uniformity test needs enough max-p samples to be stable")
    p.add_argument("--n-null", type=int, default=999,
                   help=">= required_null_draws(max_search, q) so FDR can detect a true hit")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--fdr-q", type=float, default=0.10)
    p.add_argument("--out", default="results/splicebert_partner_selection/circuits/calibration_audit.json")
    args = p.parse_args()

    # Pure-noise "head effects": no planted structure. The null draws come from the
    # same distribution, so nothing should survive selection.
    def score_fn(rng, size):
        return rng.standard_normal(size)

    def null_factory(rng, size):
        return lambda r: r.standard_normal(size)

    rows = audit_calibration(
        score_fn, null_factory,
        search_sizes=tuple(args.search_sizes), n_searches=args.n_searches,
        n_null=args.n_null, alpha=args.alpha, fdr_q=args.fdr_q, random_seed=0)

    # sizing check: with 96 tests and this q, how many null draws does a real hit need?
    need = required_null_draws(max(args.search_sizes), q=args.fdr_q, expected_hits=1)

    report = {
        "purpose": "pure-noise calibration of the max-null p-value + BH-FDR path",
        "n_null": args.n_null, "alpha": args.alpha, "fdr_q": args.fdr_q,
        "required_null_draws_for_one_hit": need,
        "n_null_sufficient": args.n_null >= need,
        "rows": rows,
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))
    ok = all(r["uniform_ok"] and r["fpr_ok"] and r["discoveries_at_fdr"] <= 0.02 for r in rows)
    print("\nCALIBRATION AUDIT:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
