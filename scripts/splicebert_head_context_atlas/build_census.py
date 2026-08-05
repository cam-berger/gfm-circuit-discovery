#!/usr/bin/env python
"""STAGE 0 (outcome-independent) — intron-mate competition cohort census + projected-MDE power gate.

Answers the red-team's gating question BEFORE any GPU spend:
  * How many USABLE competition examples exist per arm (alt-5'SS donor / alt-3'SS acceptor) and split
    (discovery vs sealed chr8/9), where "usable" = target, shared partner, AND a competitor all fit a
    1024nt window with a clean non-overlapping destroy (>= min_separation)?
  * At that realized n, what is the projected per-head MDE under 96-head multiplicity, propagated from
    the IC1/IC2 achieved precision — and does it clear a candidate SESOI-C?

Annotation only. No genome sequence, no model, no outcome. Deterministic (sorted iteration + no RNG).
Writes results/splicebert_head_context_atlas/census_v1.json + .manifest.json (refuses overwrite).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.splicing.gtf import load_transcripts  # noqa: E402
from gfmcd.splicing.competition import (  # noqa: E402
    CompetitionConfig, build_competition_index, enumerate_competition_examples,
    VAL_CHROMOSOMES, TEST_CHROMOSOMES,
)

DEFAULT_GTF: str | None = None  # supply via --gtf
STD_CHROMS = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}

# Achieved precision of the prior BEHAVIORAL competition contrasts, imported for the MDE projection.
IC1_SD_CLUSTER = 0.162   # IC1 cluster-level sd_between (nats)
IC2_SD_DID = 0.774       # IC2 beta_transport DiD SD (nats) — E_ih is a strictly higher-variance object
N_HEADS = 96             # max-T / BH multiplicity


def quantiles(xs, qs=(0.5, 0.9, 0.99)):
    if not xs:
        return {str(q): None for q in qs}
    s = sorted(xs)
    out = {}
    for q in qs:
        i = min(int(q * (len(s) - 1) + 0.5), len(s) - 1)
        out[str(q)] = s[i]
    return out


def mde(sd: float, n: float, n_tests: int = N_HEADS, alpha: float = 0.05, power: float = 0.8) -> float:
    """Two-sided per-head MDE at gene-clustered n under Bonferroni(n_tests) — an UPPER bound on the
    less-conservative max-T MDE. NaN if n<=0."""
    if n <= 0:
        return float("nan")
    # inverse normal via rational approx (Acklam); avoids a scipy dependency
    def _ppf(p):
        a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00]
        pl, ph = 0.02425, 1 - 0.02425
        if p < pl:
            q = math.sqrt(-2 * math.log(p))
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        if p <= ph:
            q = p - 0.5; r = q*q
            return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
        q = math.sqrt(-2 * math.log(1-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    z_alpha = _ppf(1 - alpha / (2 * n_tests))
    z_power = _ppf(power)
    return (z_alpha + z_power) * sd / math.sqrt(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--out", default=str(REPO / "results/splicebert_head_context_atlas/census_v1.json"))
    ap.add_argument("--min-separation", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None, help="cap transcripts (smoke only)")
    ap.add_argument("--sesoi", type=float, default=0.05, help="candidate SESOI-C (nats) for the gate")
    args = ap.parse_args()

    if os.environ.get("PYTHONHASHSEED") != "0":
        print("WARN: PYTHONHASHSEED != 0; relaunch with PYTHONHASHSEED=0 for a frozen census.", file=sys.stderr)

    cfg = CompetitionConfig(min_separation=args.min_separation)
    print(f"[census] loading protein_coding transcripts from {args.gtf} ...", file=sys.stderr)
    txs = load_transcripts(args.gtf, transcript_types=("protein_coding",), allowed_chromosomes=STD_CHROMS)
    if args.limit:
        txs = txs[: args.limit]
    print(f"[census] {len(txs)} transcripts; building intron-mate competition index ...", file=sys.stderr)
    idx = build_competition_index(txs)
    examples = enumerate_competition_examples(idx, cfg)
    print(f"[census] {len(examples)} target-level competition examples", file=sys.stderr)

    # ---- aggregate per (site_type, split) ----
    def blank():
        return {"n_targets": 0, "n_in_window": 0, "n_usable": 0, "n_tandem": 0,
                "genes": set(), "genes_usable": set(),
                "cluster_size": [], "tp_dist": [], "tc_dist": [], "span": [],
                "usable_tp_dist": [], "usable_tc_dist": [], "n_usable_competitors": []}

    agg = defaultdict(blank)
    for ex in examples:
        for key in ((ex.site_type, ex.split), ("BOTH", ex.split), (ex.site_type, "ALL")):
            a = agg[key]
            a["n_targets"] += 1
            a["genes"].add(ex.gene_id)
            a["cluster_size"].append(ex.cluster_size)
            a["tp_dist"].append(ex.target_partner_distance)
            a["tc_dist"].append(ex.nearest_competitor_distance)
            a["span"].append(ex.span)
            if ex.in_window:
                a["n_in_window"] += 1
            if ex.is_tandem:
                a["n_tandem"] += 1
            if ex.usable:
                a["n_usable"] += 1
                a["genes_usable"].add(ex.gene_id)
                a["usable_tp_dist"].append(ex.target_partner_distance)
                a["usable_tc_dist"].append(ex.target_competitor_distance)
                a["n_usable_competitors"].append(ex.n_usable_competitors)

    report = {}
    for (st, sp), a in sorted(agg.items()):
        report[f"{st}|{sp}"] = {
            "n_targets": a["n_targets"], "n_genes": len(a["genes"]),
            "n_in_window": a["n_in_window"], "n_usable": a["n_usable"],
            "n_genes_usable": len(a["genes_usable"]), "n_tandem": a["n_tandem"],
            "cluster_size_q": quantiles(a["cluster_size"]),
            "target_partner_dist_q": quantiles(a["tp_dist"]),
            "nearest_competitor_dist_q": quantiles(a["tc_dist"]),
            "span_q": quantiles(a["span"]),
            "usable_target_partner_dist_q": quantiles(a["usable_tp_dist"]),
            "usable_target_competitor_dist_q": quantiles(a["usable_tc_dist"]),
        }

    # ---- projected-MDE power gate on the SEALED usable cohort ----
    gate = {}
    for st in ("alt5ss_donor", "alt3ss_acceptor", "BOTH"):
        r = report.get(f"{st}|test", {})
        n_ex = r.get("n_usable", 0)
        n_ge = r.get("n_genes_usable", 0)
        gate[st] = {
            "sealed_usable_examples": n_ex, "sealed_usable_genes": n_ge,
            "MDE_per_head_IC1sd_genes": mde(IC1_SD_CLUSTER, n_ge),
            "MDE_per_head_IC2sd_examples": mde(IC2_SD_DID, n_ex),
            "sesoi_c_candidate": args.sesoi,
            "licensable_IC1": (mde(IC1_SD_CLUSTER, n_ge) < args.sesoi) if n_ge else False,
            "licensable_IC2": (mde(IC2_SD_DID, n_ex) < args.sesoi) if n_ex else False,
        }

    out = {
        "schema": "head_context_atlas.census/1",
        "estimand": "C — intron-mate splice competition (alt-5'SS donor + alt-3'SS acceptor)",
        "config": {"window_length": cfg.window_length, "acc_min": cfg.acc_min, "acc_max": cfg.acc_max,
                   "min_context": cfg.min_context, "min_separation": cfg.min_separation,
                   "max_span": cfg.max_span},
        "splits": {"discovery": "autosomes/other ∉ {chr8,9,11,16,19}", "val": list(VAL_CHROMOSOMES),
                   "sealed_test": list(TEST_CHROMOSOMES)},
        "n_transcripts": len(txs), "n_examples": len(examples),
        "mde_assumptions": {"IC1_SD_cluster": IC1_SD_CLUSTER, "IC2_SD_DiD": IC2_SD_DID,
                            "n_heads_multiplicity": N_HEADS, "alpha": 0.05, "power": 0.8,
                            "note": "Bonferroni upper bound on max-T MDE; E_ih is a further difference so real SD >= these anchors"},
        "census": report,
        "power_gate": gate,
    }

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.exists():
        raise FileExistsError(f"refuse to overwrite {outp} (supersede with a new artifact)")
    outp.write_text(json.dumps(out, indent=2, sort_keys=True))

    # ---- manifest ----
    try:
        from gfmcd.provenance import ExperimentManifest
        man = ExperimentManifest.create(
            experiment_id="head_context_atlas_census_v1",
            command=["python"] + sys.argv, status="EXPLORATORY",
            config=out["config"],
        )
        man.add_input(args.gtf)      # file_record computes sha256 + size
        man.add_output(str(outp))
        man.write(str(outp.with_suffix(".manifest.json")))
    except Exception as e:  # manifest is best-effort; census stands on its own
        print(f"WARN: manifest write failed: {e}", file=sys.stderr)

    # ---- human summary ----
    print("\n================ COMPETITION COHORT CENSUS (annotation-only) ================")
    for sp in ("train", "val", "test", "ALL"):
        for st in ("alt5ss_donor", "alt3ss_acceptor", "BOTH"):
            r = report.get(f"{st}|{sp}")
            if not r:
                continue
            print(f"  [{sp:5s} {st:15s}] targets={r['n_targets']:>7d} in_window={r['n_in_window']:>7d} "
                  f"USABLE={r['n_usable']:>7d} genes_usable={r['n_genes_usable']:>6d} "
                  f"tandem={r['n_tandem']:>6d}  median_intron={r['target_partner_dist_q']['0.5']}")
    print("\n---------------- SEALED (chr8/9) PROJECTED-MDE POWER GATE ----------------")
    for st, g in gate.items():
        print(f"  {st:15s}: usable_examples={g['sealed_usable_examples']:>6d} usable_genes={g['sealed_usable_genes']:>5d}")
        print(f"    MDE(IC1 sd, genes)   = {g['MDE_per_head_IC1sd_genes']:.3f} nats   vs SESOI-C {g['sesoi_c_candidate']}  -> licensable={g['licensable_IC1']}")
        print(f"    MDE(IC2 sd, examples)= {g['MDE_per_head_IC2sd_examples']:.3f} nats   vs SESOI-C {g['sesoi_c_candidate']}  -> licensable={g['licensable_IC2']}")
    print(f"\nWrote {outp}")


if __name__ == "__main__":
    main()
