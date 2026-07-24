#!/usr/bin/env python
"""IC2 v5.2 pre-freeze calibration (TRAIN-only, resource-capped). Cam-directed order:
  Part 1  D_m FEASIBILITY: match a decoy to each scaffold P on the model's single-candidate f-drop measured
          in a FROZEN INDEPENDENT reference panel (Delta_f_ref). Report yield, match residual, decoy-source pool.
  Part 2  CROSS-FITTED DISCRIMINABILITY: TRAIN-A freezes a motif-divergent reliance-matched selection RULE;
          disjoint TRAIN-B tests whether that frozen control yields |beta_transport| >= 0.05 (the wording gate).
          beta_transport = [S(P)-S(D)]_displaced - [S(P)-S(D)]_native = tau_D - tau_P.
  Part 3  ICC / DESIGN-EFFECT: variance of tau_D across decoy-SOURCE genes -> design effect -> cluster floor.
Does NOT read val. Determinism: sorted(set()) + pinned PYTHONHASHSEED upstream; rng seeded.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts" / "splicebert"))
from gfmcd.splicing.circuit import (                                             # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, load_adapter, pick_device, eligible, _dataset)
from gfmcd.splicing.coupling import substitute                                    # noqa: E402
from run_ic2_feasibility import logits_at_donor                                   # noqa: E402


def gh(g):  # deterministic gene hash (independent of PYTHONHASHSEED)
    return int(hashlib.sha256(str(g).encode()).hexdigest()[:8], 16)


def comp_vec(w):
    """composition/motif descriptor for divergence: GC frac, pyrimidine frac, and 2-mer profile (sorted keys)."""
    w = w.upper(); n = len(w)
    gc = sum(c in "GC" for c in w) / n
    pyr = sum(c in "CT" for c in w) / n
    ks = sorted({a + b for a in "ACGT" for b in "ACGT"})
    di = np.array([w.count(k) for k in ks], float); di /= max(di.sum(), 1)
    return np.concatenate([[gc, pyr], di])


def f_at(ad, task, seqs, donors, batch):
    return [r[0] for r in logits_at_donor(ad, task, seqs, donors, batch)]  # sealed max-margin only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pool", type=int, default=400)        # windows scored for Delta_f_ref (one per gene)
    ap.add_argument("--n-scaffold", type=int, default=120)    # scaffolds for Parts 2/3
    ap.add_argument("--panel", type=int, default=3)           # frozen reference-scaffold panel size
    ap.add_argument("--match-tol", type=float, default=0.05)  # |Delta_f_ref(D)-Delta_f_ref(P)| tolerance
    ap.add_argument("--reuse-cap", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF); ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="results/splicebert_coupling/development/ic2_calibration.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    dev = pick_device(args.device); ad = load_adapter(args.model, dev); task = CouplingTask(ad, batch=args.batch)
    rng = np.random.default_rng(0)

    tr = eligible(_dataset(args.gtf, args.fasta), "train")
    seen, dd = set(), []
    for x in tr:
        if x.gene_id not in seen and (x.next_acceptor_pos - x.donor_pos) >= 110 and x.next_acceptor_pos - 37 >= 0:
            seen.add(x.gene_id); dd.append(x)
    dd = sorted(dd, key=lambda x: gh(x.gene_id))              # deterministic order
    def win(x): return x.sequence[x.next_acceptor_pos - 37:x.next_acceptor_pos + 3]
    dd = [x for x in dd if len(win(x)) == 40]
    # frozen reference panel: deterministic scaffolds at length quantiles (independent context for Delta_f_ref)
    by_len = sorted(dd, key=lambda x: (x.next_acceptor_pos - x.donor_pos, gh(x.gene_id)))
    qs = np.linspace(0.2, 0.8, args.panel)
    panel = [by_len[int(q * (len(by_len) - 1))] for q in qs]
    panel_ids = {p.gene_id for p in panel}
    pool = [x for x in dd if x.gene_id not in panel_ids][:args.n_pool]
    # TRAIN-A / TRAIN-B split by gene hash parity
    A = [x for x in pool if gh(x.gene_id) % 2 == 0]; B = [x for x in pool if gh(x.gene_id) % 2 == 1]
    print(f"eligible dedup={len(dd)} panel={len(panel)} pool={len(pool)} TRAIN-A={len(A)} TRAIN-B={len(B)}")

    # ---- Delta_f_ref(window): single-candidate reliance averaged over the frozen reference panel ----
    pool_win = [win(x) for x in pool]
    def delta_f_ref_full(windows):
        seqs, dons, key = [], [], []
        for wi, w in enumerate(windows):
            for ri, R in enumerate(panel):
                a = R.next_acceptor_pos
                base = R.sequence[:a - 37] + w + R.sequence[a + 3:]
                seqs += [base, substitute(base, a - 2, "CC")]; dons += [R.donor_pos, R.donor_pos]
                key += [(wi, ri, "i"), (wi, ri, "c")]
        fs = f_at(ad, task, seqs, dons, args.batch)
        agg = {}
        for (wi, ri, t), v in zip(key, fs):
            agg.setdefault((wi, ri), {})[t] = v
        out = []
        for wi in range(len(windows)):
            out.append(float(np.mean([agg[(wi, ri)]["i"] - agg[(wi, ri)]["c"] for ri in range(len(panel))])))
        return np.array(out)
    dref = delta_f_ref_full(pool_win)
    comp = np.array([comp_vec(w) for w in pool_win])
    idx = {x.gene_id: i for i, x in enumerate(pool)}

    # ---- Part 1: D_m feasibility ----
    matches = 0; residuals = []; sources = set(); reuse = {}
    order = sorted(range(len(pool)), key=lambda i: gh(pool[i].gene_id))
    for i in order:
        cand = [j for j in range(len(pool)) if j != i and pool[j].gene_id != pool[i].gene_id
                and abs(dref[j] - dref[i]) <= args.match_tol and reuse.get(pool[j].gene_id, 0) < args.reuse_cap]
        if cand:
            j = min(cand, key=lambda j: (abs(dref[j] - dref[i]), gh(pool[j].gene_id)))
            matches += 1; residuals.append(abs(dref[j] - dref[i])); sources.add(pool[j].gene_id)
            reuse[pool[j].gene_id] = reuse.get(pool[j].gene_id, 0) + 1
    part1 = {"yield_frac": matches / len(pool), "n_scaffold": len(pool),
             "match_residual_median": float(np.median(residuals)) if residuals else None,
             "match_residual_p90": float(np.percentile(residuals, 90)) if residuals else None,
             "n_independent_decoy_sources": len(sources),
             "max_reuse": max(reuse.values()) if reuse else 0,
             "dref_range": [float(dref.min()), float(dref.max())], "dref_sd": float(dref.std())}

    # ---- reciprocal-swap beta_transport for a (P_gene, D_gene) pair at native/displaced slots ----
    def beta_transport_batch(pairs):
        """pairs = [(Px, Dwin)]; returns beta_transport = tau_D - tau_P per pair."""
        seqs, dons, key = [], [], []
        for pi, (Px, Dw) in enumerate(pairs):
            a, d0 = Px.next_acceptor_pos, Px.donor_pos; aD = a - 60; P = win(Px)
            def b2(wa, wb, cA, cB):
                s = Px.sequence[:aD - 37] + wb + Px.sequence[aD + 3:]
                s = s[:a - 37] + wa + s[a + 3:]
                if cA: s = substitute(s, a - 2, "CC")
                if cB: s = substitute(s, aD - 2, "CC")
                return s
            # config N (P@nat,D@dis): need 00,A0,0B ; config W (D@nat,P@dis): 00,A0,0B
            for cn, (wa, wb) in {"N": (P, Dw), "W": (Dw, P)}.items():
                for cA, cB, sfx in [(0, 0, "00"), (1, 0, "A0"), (0, 1, "0B")]:
                    seqs.append(b2(wa, wb, cA, cB)); dons.append(d0); key.append((pi, cn + sfx))
        fs = f_at(ad, task, seqs, dons, args.batch)
        agg = {}
        for (pi, nm), v in zip(key, fs):
            agg.setdefault(pi, {})[nm] = v
        out = []
        for pi in range(len(pairs)):
            g = agg[pi]
            # S(X,slot)=reliance; slot_a=native, slot_b=displaced
            SP_nat = g["N00"] - g["NA0"]; SD_dis = g["N00"] - g["N0B"]
            SD_nat = g["W00"] - g["WA0"]; SP_dis = g["W00"] - g["W0B"]
            tau_P = SP_nat - SP_dis; tau_D = SD_nat - SD_dis
            out.append((tau_D - tau_P, tau_P, tau_D))            # beta_transport, tau_P, tau_D
        return out

    # ---- Part 2: cross-fitted discriminability ----
    # TRAIN-A: FREEZE the rule = "reliance-matched (|dDref|<=tol) AND composition-divergence in TRAIN-A's top tercile".
    def matched_divergent(scaffolds, div_thresh):
        pairs = []
        for x in scaffolds:
            i = idx[x.gene_id]
            cand = [j for j in range(len(pool)) if pool[j].gene_id != x.gene_id
                    and abs(dref[j] - dref[i]) <= args.match_tol]
            if not cand: continue
            j = max(cand, key=lambda j: (np.linalg.norm(comp[j] - comp[i]), -gh(pool[j].gene_id)))  # most divergent matched
            dvg = float(np.linalg.norm(comp[j] - comp[i]))
            if div_thresh is None or dvg >= div_thresh:
                pairs.append((x, pool_win[j], dvg))
        return pairs
    # discover divergence distribution on TRAIN-A, freeze threshold = TRAIN-A top-tercile (66th pct)
    a_pairs_all = matched_divergent(A, None)
    div_thresh = float(np.percentile([d for *_, d in a_pairs_all], 66)) if a_pairs_all else 0.0
    # apply FROZEN rule to TRAIN-B, measure beta_transport
    b_pairs = matched_divergent(B[:args.n_scaffold], div_thresh)
    bt = beta_transport_batch([(x, w) for x, w, _ in b_pairs]) if b_pairs else []
    bt_arr = np.array([t[0] for t in bt]) if bt else np.array([])
    part2 = {"train_a_divergence_thresh_66pct": div_thresh, "n_train_b_divergent_pairs": len(b_pairs),
             "beta_transport_mean": float(bt_arr.mean()) if len(bt_arr) else None,
             "beta_transport_absmean": float(np.abs(bt_arr).mean()) if len(bt_arr) else None,
             "beta_transport_sd": float(bt_arr.std(ddof=1)) if len(bt_arr) > 1 else None,
             "frac_abs_ge_0.05": float(np.mean(np.abs(bt_arr) >= 0.05)) if len(bt_arr) else None,
             "DISCRIMINABILITY_CONFIRMED": bool(len(bt_arr) and abs(bt_arr.mean()) >= 0.05)}

    # ---- Part 3: ICC of tau_D across decoy-source genes (reliance-matched D_m, reuse-capped) ----
    dm_pairs, dm_src = [], []
    reuse2 = {}
    for x in B[:args.n_scaffold]:
        i = idx[x.gene_id]
        cand = [j for j in range(len(pool)) if pool[j].gene_id != x.gene_id
                and abs(dref[j] - dref[i]) <= args.match_tol and reuse2.get(pool[j].gene_id, 0) < args.reuse_cap]
        if not cand: continue
        j = min(cand, key=lambda j: (abs(dref[j] - dref[i]), gh(pool[j].gene_id)))
        reuse2[pool[j].gene_id] = reuse2.get(pool[j].gene_id, 0) + 1
        dm_pairs.append((x, pool_win[j])); dm_src.append(pool[j].gene_id)
    dm_bt = beta_transport_batch(dm_pairs) if dm_pairs else []
    tauD = np.array([t[2] for t in dm_bt]); btD = np.array([t[0] for t in dm_bt])
    # one-way ANOVA ICC of tau_D by decoy-source gene
    def icc(vals, groups):
        vals = np.asarray(vals); gs = sorted(set(groups))
        if len(gs) < 2: return None
        gm = {g: vals[[k for k in range(len(groups)) if groups[k] == g]] for g in gs}
        k = len(gs); N = len(vals); nbar = N / k
        grand = vals.mean()
        ssb = sum(len(v) * (v.mean() - grand) ** 2 for v in gm.values())
        ssw = sum(((v - v.mean()) ** 2).sum() for v in gm.values())
        msb = ssb / (k - 1); msw = ssw / max(N - k, 1)
        icc_v = (msb - msw) / (msb + (nbar - 1) * msw) if (msb + (nbar - 1) * msw) > 0 else 0.0
        return float(max(icc_v, 0.0)), k, float(nbar), float(msw)
    ic = icc(tauD, dm_src)
    part3 = {"n_pairs": len(dm_pairs), "n_decoy_source_clusters": len(set(dm_src)),
             "tauD_mean": float(tauD.mean()) if len(tauD) else None,
             "beta_transport_matched_mean": float(btD.mean()) if len(btD) else None,
             "beta_transport_matched_sd": float(btD.std(ddof=1)) if len(btD) > 1 else None}
    if ic:
        iccv, k, nbar, msw = ic
        deff = 1 + (nbar - 1) * iccv
        part3.update({"tauD_ICC_decoy_source": iccv, "design_effect": float(deff),
                      "n_clusters": k, "avg_cluster_size": nbar})

    rep = {"part1_Dm_feasibility": part1, "part2_cross_fitted_discriminability": part2,
           "part3_ICC_design_effect": part3, "config": vars(args)}
    out = REPO / args.out; out.write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float)); print(f"saved -> {out}")


if __name__ == "__main__":
    main()
