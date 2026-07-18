"""Caduceus-PS T1 runner: load constructs + model -> preflight/memorials -> tau (discovery) -> decision (confirm).
Works with the real Caduceus (CUDA) OR the MockCaduceus (local validation) via --mock.
  python run_t1.py --constructs cohort_constructs.jsonl [--mock] [--limit N] --out results_t1.json
Only the forward passes need CUDA; everything else (tau, stats, decision) is model-agnostic.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
import numpy as np
from transformers import AutoTokenizer
import caduceus_t1 as H

MID = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"


def load(path, limit=0):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows[:limit] if limit else rows


def all_margins(scorer, rows):
    """One flat pass over every (construct) across all examples; equal-length (1024nt) so batched cleanly."""
    seqs, pos, key = [], [], []
    for i, r in enumerate(rows):
        for tag, (s, p) in r["constructs"].items():
            seqs.append(s); pos.append(p); key.append((i, tag))
    m = scorer.donor_margin(seqs, pos)
    by = defaultdict(dict)
    for (i, tag), v in zip(key, m):
        by[i][tag] = float(v)
    return by


def tau_permuted_null(by, rows, seed=0):
    """tau = upper 95% CI of rho_partner under the PERMUTED-partner null on DISCOVERY (swap partner<->nonpartner).
    Non-circular, within-model (fork a default)."""
    idx = [i for i, r in enumerate(rows) if r["split"] == "discovery"]
    genes = [rows[i]["gene"] for i in idx]
    perm_ps, A = [], []
    for i in idx:
        b = by[i]; clean = b["clean"]
        C = clean - b["partner_cc"]; C_np = clean - b["nonpartner_cc"]
        perm_ps.append(C_np - C)                     # swapped labels => null partner-specificity
        A.append(clean - b["donor_dead"])
    r = H.paired_gene_bootstrap_ratio(perm_ps, A, genes, seed=seed)
    return max(0.0, r["ci"][1])                      # upper CI of the null ratio


def lr_curve(by, rows, seed=0):
    """Gene-clustered CI of |Delta margin| at each downstream block offset (long-range transmission)."""
    offs = sorted({int(t.split("@")[1]) for r in rows for t in r["constructs"] if t.startswith("lr_block@")})
    out = {}
    for off in offs:
        vals, genes = [], []
        for i, r in enumerate(rows):
            t = f"lr_block@{off}"
            if t in by[i] and "clean" in by[i]:
                vals.append(abs(by[i]["clean"] - by[i][t])); genes.append(r["gene"])
        if vals:
            gm, _ = H.gene_means(vals, genes)
            out[off] = H.bca_ci(gm, seed=seed)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--constructs", default="cohort_constructs.jsonl")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="results_t1.json")
    args = ap.parse_args()

    rows = load(args.constructs, args.limit)
    tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
    bid = {b: tok.convert_tokens_to_ids(b) for b in "ACGT"}
    if args.mock:
        from mock_caduceus import MockCaduceus
        model, dev = MockCaduceus(base_ids=bid), "cpu"
    else:
        import torch
        from transformers import AutoModelForMaskedLM
        model = AutoModelForMaskedLM.from_pretrained(MID, trust_remote_code=True)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(dev)
    from caduceus_scorer import CaduceusScorer
    sc = CaduceusScorer(model, tok, device=dev, batch_size=args.batch)

    # --- preflight (memorials on the real model; scorer.__init__ already asserted tokenization/coords) ---
    mlm = sc.mlm_health([r["constructs"]["clean"][0] for r in rows[:20]])
    by = all_margins(sc, rows)

    tau = tau_permuted_null(by, rows)
    curve = lr_curve(by, rows)

    genes = [r["gene"] for r in rows]; splits = [r["split"] for r in rows]
    ex_vals = defaultdict(list)
    for i, r in enumerate(rows):
        for tag in ("clean", "partner_cc", "donor_dead", "nonpartner_cc", "clean_other", "partner_cc_other"):
            ex_vals[tag].append(by[i][tag])
    res = H.decide(dict(ex_vals), genes, splits, tau=tau, lr_curve=curve, mlm_health=mlm)
    res["tau_source"] = "permuted-partner-null upper CI (discovery)"
    res["long_range_curve"] = {str(k): v for k, v in curve.items()}
    res["n_examples"] = len(rows); res["mock"] = args.mock
    json.dump(res, open(args.out, "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("VERDICT", "gates", "rho_partner", "tau", "C_ci",
                                          "partner_spec_ci", "joint_power", "long_range_certified")}, indent=1, default=float))
    print(f"saved -> {args.out}  (mock={args.mock})")


if __name__ == "__main__":
    main()
