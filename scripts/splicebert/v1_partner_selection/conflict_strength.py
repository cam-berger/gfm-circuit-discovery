#!/usr/bin/env python
"""Characterize CONFLICT STRENGTH across the local-conflict set.

A "conflict" is an example where the PWM+length baseline's arg-max is a decoy, not
the true pair (see `baselines.baseline_conflict_examples`). But conflicts are not
all equal: the baseline may prefer its decoy by a hair or by a mile. The strength is

    conflict_strength = max(decoy_scores) - true_score  =  -baseline.true_margin(x)   (> 0 on a conflict)

in baseline log-score units (acceptor_pwm + donor_pwm + length_weight*length_prior).

This script quantifies that distribution so the write-up can state honestly HOW hard
the conflicts are, rather than treating "conflict" as a binary. It reads only the
frozen conflict set + the frozen baseline pickle; it trains nothing and touches no
model. Emits a manifest-backed JSON + a text histogram.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import pickle  # noqa: E402
from gfmcd.splicing.io import load_jsonl  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--conflicts", default="colab_results/splicebert_run/baselines/val_local_conflicts.jsonl")
    p.add_argument("--baseline", default="colab_results/splicebert_run/baselines/local_pair_baseline.pkl")
    p.add_argument("--out", default="results/splicebert_partner_selection/behavior/conflict_strength.json")
    args = p.parse_args()

    conflicts_path = REPO / args.conflicts
    baseline_path = REPO / args.baseline

    with open(baseline_path, "rb") as f:
        baseline = pickle.load(f)
    examples = load_jsonl(conflicts_path, validate=False)

    strengths, margins = [], []
    n_true_wins = 0  # sanity: should be ~0 — every example here is meant to be a conflict
    for x in examples:
        m = baseline.true_margin(x)          # true - best_decoy ; < 0 on a real conflict
        margins.append(m)
        if m >= 0:
            n_true_wins += 1
            continue
        strengths.append(-m)                 # conflict strength > 0

    s = np.asarray(strengths, dtype=np.float64)
    qs = [0, 5, 10, 25, 50, 75, 90, 95, 100]
    pct = {f"p{q}": float(np.percentile(s, q)) for q in qs}

    # "weak" conflicts: baseline barely prefers the decoy. Report the fraction under a
    # few thresholds in score units so the write-up can pick a defensible cutoff.
    thresholds = [0.1, 0.25, 0.5, 1.0, 2.0]
    frac_below = {f"<{t}": float(np.mean(s < t)) for t in thresholds}

    # text histogram (20 bins over [0, p99]) for eyeballing
    hi = float(np.percentile(s, 99))
    counts, edges = np.histogram(s, bins=20, range=(0.0, hi))
    width = max(1, counts.max())
    hist_lines = [
        f"  [{edges[i]:6.2f},{edges[i+1]:6.2f})  {'#' * int(40 * counts[i] / width):40s} {counts[i]}"
        for i in range(len(counts))
    ]

    report = {
        "purpose": "distribution of baseline decoy-preference (conflict strength) over the local-conflict set",
        "units": "baseline log-score (acceptor_pwm + donor_pwm + length_weight*length_prior)",
        "n_conflicts": len(examples),
        "n_true_wins_unexpected": n_true_wins,
        "conflict_strength": {
            "mean": float(s.mean()),
            "median": float(np.median(s)),
            "std": float(s.std()),
            "percentiles": pct,
            "frac_below_threshold": frac_below,
        },
        "provenance": {
            "conflicts_file": str(conflicts_path.relative_to(REPO)),
            "conflicts_sha256": sha256(conflicts_path),
            "baseline_file": str(baseline_path.relative_to(REPO)),
            "baseline_sha256": sha256(baseline_path),
        },
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}, indent=2))
    print("\nconflict-strength histogram (0 .. p99):")
    print("\n".join(hist_lines))
    print(f"\nsaved -> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
