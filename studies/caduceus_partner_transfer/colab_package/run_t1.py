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


def tau_permuted_null(by, rows, seeds=(0, 1, 2, 17)):
    """tau = GENUINE permuted-partner-null floor on DISCOVERY: relabel one non-partner AG as the nominal
    'partner' and a SECOND non-partner as its reference, so the contrast contains NO real partner identity
    (null_ps = C_nonpartner1 - C_nonpartner2). tau = upper 95% CI of |null rho_partner|, taken as the MAX over
    independently-seeded bootstraps (R22 sampling-stability guard). Non-circular; independent of the alternative."""
    idx = [i for i, r in enumerate(rows) if r["split"] == "discovery" and "nonpartner2_cc" in by[i]]
    if not idx:
        raise RuntimeError("no discovery examples carry a 2nd non-partner (nonpartner2_cc) -> cannot seal a genuine tau")
    genes = [rows[i]["gene"] for i in idx]
    null_ps, A = [], []
    for i in idx:
        b = by[i]; clean = b["clean"]
        null_ps.append((clean - b["nonpartner_cc"]) - (clean - b["nonpartner2_cc"]))  # both non-partners => null
        A.append(clean - b["donor_dead"])
    ups = []
    for s in seeds:
        r = H.paired_gene_bootstrap_ratio(null_ps, A, genes, seed=s)
        ups.append(max(0.0, r["ci"][1]))
    return float(max(ups)), {"n_discovery_null": len(idx), "tau_across_seeds": [float(u) for u in ups]}


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

    # --- HARD PREFLIGHT memorials (STOP before the full pass; §3 on-path, can fail) ---
    sub = rows[:64]
    seq0, d0 = sub[0]["constructs"]["clean"]
    m_d = float(sc.donor_margin([seq0], [d0])[0])
    m_p1 = float(sc.donor_margin([seq0], [d0 + 1])[0]); m_m1 = float(sc.donor_margin([seq0], [d0 - 1])[0])
    mlm = sc.mlm_health([r["constructs"]["clean"][0] for r in sub[:20]])
    cl = sc.donor_margin([r["constructs"]["clean"][0] for r in sub], [r["constructs"]["clean"][1] for r in sub])
    dd = sc.donor_margin([r["constructs"]["donor_dead"][0] for r in sub], [r["constructs"]["donor_dead"][1] for r in sub])
    A_sub = float(np.mean(cl - dd))
    minus_share = np.mean([r.get("strand") == "-" for r in rows])
    preflight = {"margin_donor_clean": m_d, "coord_sensitive_dp1": m_d > m_p1, "coord_sensitive_dm1": m_d > m_m1,
                 "mlm_health": float(mlm), "A_donor_sub": A_sub, "minus_strand_share": float(minus_share)}
    fails = []
    if not args.mock:
        if not (m_d > 0): fails.append("donor not recognised")
        if not (m_d > m_p1 and m_d > m_m1): fails.append("coordinate off-by-one (readout not at donor, both ±1)")
        if not (mlm > 0): fails.append("mlm_health<=0")
        if not (A_sub > 0): fails.append("A_donor<=0")
        if not (0.3 <= minus_share <= 0.7): fails.append(f"both_strands_present out of [0.3,0.7]: {minus_share:.3f}")
        if fails:
            raise RuntimeError("PREFLIGHT MEMORIAL FAILURE (no GPU wasted on the full pass): " + "; ".join(fails))

    by = all_margins(sc, rows)
    tau, tau_meta = tau_permuted_null(by, rows)
    curve = lr_curve(by, rows)

    genes = [r["gene"] for r in rows]; splits = [r["split"] for r in rows]
    ex_vals = defaultdict(list)
    for i, r in enumerate(rows):
        for tag in ("clean", "partner_cc", "donor_dead", "nonpartner_cc", "clean_other", "partner_cc_other"):
            ex_vals[tag].append(by[i][tag])
    res = H.decide(dict(ex_vals), genes, splits, tau=tau, lr_curve=curve, mlm_health=mlm,
                   both_strands_ok=bool(0.3 <= minus_share <= 0.7))
    res["preflight"] = preflight; res["tau_meta"] = tau_meta
    res["tau_source"] = "GENUINE permuted-partner null (2 non-partners), max over seeds {0,1,2,17} (discovery)"
    res["long_range_curve"] = {str(k): v for k, v in curve.items()}
    res["n_examples"] = len(rows); res["mock"] = args.mock
    json.dump(res, open(args.out, "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("VERDICT", "gates", "rho_partner", "tau", "C_ci",
                                          "partner_spec_ci", "joint_power", "long_range_certified")}, indent=1, default=float))
    print(f"saved -> {args.out}  (mock={args.mock})")


if __name__ == "__main__":
    main()
