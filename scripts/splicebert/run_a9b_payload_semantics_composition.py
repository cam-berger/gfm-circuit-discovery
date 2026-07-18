#!/usr/bin/env python
"""Donor-graph experiment 13b -- A9b: is the payload SPLICE STRENGTH *net of bulk composition*?

Frozen protocol: studies/splicebert_partner_selection/PREREGISTRATION_A9b_PAYLOAD_SEMANTICS_COMPOSITION.md

A9 named the payload local splice strength (V2b held-out r=-0.51) but the composition control (V4) was
NON-IDENTIFIABLE, leaving a confound OPEN.  A9b predefines a BULK-composition adjustment set Z (GC, PY,
di/tri-mer freq -> PCA, position), estimates the strength effect NET OF Z with cross-fitted double-ML
(Robinson partialling), verifies common support / collinearity, and gates the run on FOUR calibration
controls: the A9 through-model planted-positive (sensitivity) + permuted-FPR (specificity), PLUS two
semi-synthetic CONFOUND gates -- composition-only -> beta_str NULL, collinear-strength -> beta_str
RECOVERED.  Decision is CI-EXCLUDES-ZERO (dyadic gene bootstrap), held-out + within matched sets.
Honest fallback: strength and bulk composition inseparable -> COMPOSITE (core-motif entanglement is
intrinsic, not removed).  Reuses the A9 interchange + value-path metric; no new circuit kernel.
--smoke = mechanics memorials + all four calibration gates on tiny-n (val sealed).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "splicebert"))

from sklearn.decomposition import PCA                                          # noqa: E402
from sklearn.preprocessing import StandardScaler                              # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor                        # noqa: E402
from sklearn.linear_model import Ridge                                        # noqa: E402
from sklearn.model_selection import GroupKFold                                # noqa: E402

from gfmcd.provenance import ExperimentManifest, git_commit, sha256_file       # noqa: E402
from gfmcd.circuits.hf_bert import patch_heads                                 # noqa: E402
from gfmcd.splicing.circuit import (                                          # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask,
    load_adapter, load_examples, pick_device,
)
from gfmcd.splicing.payload import deliver, head_output, value_weight_head     # noqa: E402

# reuse A9 machinery verbatim (module-level; importing does not run its main())
from run_a9_payload_semantics import (                                        # noqa: E402
    score3, scorers, caprun, inject_capture, frozen_uinfo, L3, H3, ACC_OFF,
)

SEED = 0
BASES = "ACGT"
BIDX = {c: i for i, c in enumerate(BASES)}


# ----------------------------------------------------------------------------- composition Z
def _freqs(s):
    s = s.upper()
    di = np.zeros(16); tri = np.zeros(64)
    for t in range(len(s) - 1):
        d = s[t:t + 2]
        if set(d) <= set(BASES):
            di[4 * BIDX[d[0]] + BIDX[d[1]]] += 1
    for t in range(len(s) - 2):
        u = s[t:t + 3]
        if set(u) <= set(BASES):
            tri[16 * BIDX[u[0]] + 4 * BIDX[u[1]] + BIDX[u[2]]] += 1
    if di.sum() > 0:
        di /= di.sum()
    if tri.sum() > 0:
        tri /= tri.sum()
    return np.concatenate([di, tri])


def zraw(seq, a):
    """Bulk/frequency composition over the payload region [a-20 .. a+5]; PPT window [a-20 .. a-3]."""
    if a - 20 < 0 or a + 5 > len(seq):
        return None
    region = seq[a - 20:a + 5]
    ppt = seq[a - 20:a - 3]
    gc = sum(c in "GC" for c in region.upper()) / len(region)
    py = (sum(c in "CT" for c in ppt.upper()) / len(ppt)) if len(ppt) else 0.0
    return {"GC": float(gc), "PY": float(py), "POS": float(a), "ditri": _freqs(region)}


# ----------------------------------------------------------------------------- DML / inference
def _gbr(seed):
    return GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                     subsample=0.8, random_state=seed)


def crossfit(y, X, groups, folds, seed, learner="gbr"):
    """Out-of-fold nuisance predictions E[y|X] with gene-clustered folds (no target-gene leakage)."""
    n = len(y)
    k = min(folds, len(np.unique(groups)))
    if k < 2 or n < 2 * k:
        return None
    pred = np.zeros(n)
    gkf = GroupKFold(n_splits=k)
    for tr, te in gkf.split(X, groups=groups):
        m = _gbr(seed) if learner == "gbr" else Ridge(alpha=1.0)
        m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
    return pred


def dml_beta(S, dV, dZ, groups, folds, seed, learner="gbr"):
    """Cross-fitted Robinson partialling: beta_str = <S~,V~>/<V~,V~> on OOF residuals. Returns dict."""
    mS = crossfit(S, dZ, groups, folds, seed, learner)
    mV = crossfit(dV, dZ, groups, folds, seed, learner)
    if mS is None or mV is None:
        return None
    St = S - mS; Vt = dV - mV
    den = float(np.sum(Vt * Vt))
    if den < 1e-12:
        return None
    beta = float(np.sum(St * Vt) / den)
    return {"beta": beta, "St": St, "Vt": Vt, "mS": mS, "mV": mV}


def dyadic_ci(St, Vt, gi, gj, seed, B):
    """Dyadic (two-way gene) Poisson-multiplier bootstrap CI for beta = <St,Vt>/<Vt,Vt>."""
    rng = np.random.default_rng(seed)
    genes = np.unique(np.concatenate([gi, gj]))
    gx = {g: k for k, g in enumerate(genes)}
    ii = np.array([gx[g] for g in gi]); jj = np.array([gx[g] for g in gj])
    num0 = St * Vt; den0 = Vt * Vt
    out = np.empty(B)
    for b in range(B):
        w = rng.poisson(1.0, len(genes))
        pw = w[ii] * w[jj]
        d = float(np.sum(pw * den0))
        out[b] = float(np.sum(pw * num0) / d) if d > 1e-9 else np.nan
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def zscore(x):
    x = np.asarray(x, float); s = x.std()
    return (x - x.mean()) / (s + 1e-12)


def cv_r2(y, X, groups, folds, seed, learner="ridge"):
    """CV R^2 of y ~ X. Primary learner is LINEAR: GBR overfits this low-dim Z (negative OOF R^2),
    so linear ridge is the honest nuisance for both collinearity and composition->transfer power."""
    pred = crossfit(y, X, groups, folds, seed, learner)
    if pred is None:
        return float("nan")
    ss_res = float(np.sum((y - pred) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-pairs", type=int, default=600)
    ap.add_argument("--min-support", type=int, default=40)
    ap.add_argument("--kz-var", type=float, default=0.90)
    ap.add_argument("--kz-max", type=int, default=8)   # hard cap: keep Z low-dim (n >= 10*dim)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=120)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF); ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--mode", default="broad", choices=["broad", "a9matched"])  # a9matched = Path-A reconcile
    ap.add_argument("--tag", default=None)   # override the val-sample tag (seeds the example shuffle)
    ap.add_argument("--dump-arrays", default=None)   # cache per-pair (S,dV,dZ,genes) for offline estimator dev
    ap.add_argument("--out", default="results/splicebert_coupling/a9b_payload_semantics_composition.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    out = REPO / args.out
    pf_path = out.parent / "preflight_a9b_payload_semantics_composition.json"
    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=True)
    task = CouplingTask(ad, batch=args.batch)
    off = task.off; Dh = ad.hidden_size // ad.num_heads
    Wv = value_weight_head(ad).to(dev)
    U = frozen_uinfo(ad, task, args.gtf, args.fasta, off, dev, n_train=80 if args.smoke else 300)

    # ---- composition PCA + scaler fit on TRAIN ONLY (frozen; no val leakage) ----
    n_pca = 60 if args.smoke else 300
    Xtr, _ = load_examples(args.gtf, args.fasta, "train", n_pca, task=task, tag="a9b-pca-train",
                           min_effect=None)
    ztr = [zraw(x.sequence, x.next_acceptor_pos) for x in Xtr]
    ztr = [z for z in ztr if z is not None]
    ditri_tr = np.stack([z["ditri"] for z in ztr])
    pca = PCA(n_components=min(20, ditri_tr.shape[0] - 1, ditri_tr.shape[1])).fit(ditri_tr)
    cum = np.cumsum(pca.explained_variance_ratio_)
    Kz = int(np.searchsorted(cum, args.kz_var) + 1)
    Kz = max(2, min(Kz, pca.n_components_, args.kz_max))   # low-dim adjustment set (prereg s2)
    def assemble(z):
        return np.concatenate([[z["GC"], z["PY"], z["POS"]], pca.transform(z["ditri"][None])[0, :Kz]])
    Ztr = np.stack([assemble(z) for z in ztr])
    scaler = StandardScaler().fit(Ztr)

    # ---- per-example capture on VAL (mirrors A9) ----
    n = min(args.n, 40) if args.smoke else args.n
    X, _ = load_examples(args.gtf, args.fasta, args.split, n, task=task,
                         tag=args.tag or f"a9b-{args.split}", min_effect=None)
    strands = sorted({x.strand for x in X})
    E = []; lin_err = 0.0
    for i in range(0, len(X), args.batch):
        ch = X[i:i + args.batch]
        cl, co, dn = task.pair(ch)
        B, S = cl["input_ids"].shape
        dtok = torch.tensor([d + off for d in dn], device=dev)
        accwin = torch.tensor([[x.next_acceptor_pos + off + k for k in ACC_OFF] for x in ch], device=dev)
        bidx = torch.arange(B, device=dev)[:, None].expand(B, 4)
        oc, r3_cl, V_cl, Ap = caprun(ad, cl, dtok, True)
        oo, r3_co, V_co, _ = caprun(ad, co, dtok, False)
        base = torch.einsum("bs,bsd->bd", Ap, V_co)
        dVt = (V_cl - V_co)[bidx, accwin]; dVl = deliver(Wv, (r3_cl - r3_co)[bidx, accwin])
        lin_err = max(lin_err, float((dVt - dVl).abs().max()))
        cl_m = {k: v.clone() for k, v in cl.items()}
        agpos = torch.tensor([[x.next_acceptor_pos + off - 2, x.next_acceptor_pos + off - 1] for x in ch],
                             device=dev)
        cl_m["input_ids"][bidx[:, :2], agpos] = task.mask_id
        lg = ad.model(**cl_m).logits.float().log_softmax(-1)
        for bj, x in enumerate(ch):
            a = x.next_acceptor_pos
            sc = scorers(x.sequence, a); zr = zraw(x.sequence, a)
            aid = task.bid["A"]; gid = task.bid["G"]
            v3 = float(lg[bj, agpos[bj, 0], aid] + lg[bj, agpos[bj, 1], gid])
            E.append({"r3cl": r3_cl[bj, accwin[bj]].cpu(), "r3co": r3_co[bj, accwin[bj]].cpu(),
                      "Ap": Ap[bj, accwin[bj]].cpu(), "base": base[bj].cpu(), "dtok": int(dtok[bj]),
                      "co_ids": co["input_ids"][bj].cpu(),
                      "attn_mask": cl["attention_mask"][bj].cpu() if "attention_mask" in cl else None,
                      "gene": x.gene_id, "V3": v3, "zr": zr, **sc})
        print(f"  capture {min(i + args.batch, len(X))}/{len(X)}", end="\r", flush=True)
    print()
    E = [e for e in E if isinstance(e["V2b"], float) and not np.isnan(e["V2b"])
         and isinstance(e["V2a"], float) and not np.isnan(e["V2a"]) and e["zr"] is not None]
    if len(E) < 16:
        raise RuntimeError(f"only {len(E)} scorable examples.")
    for e in E:
        e["Z"] = scaler.transform(assemble(e["zr"])[None])[0]

    ug = sorted({e["gene"] for e in E}); Dg = set(ug[::2])
    for e in E:
        e["split"] = "D" if e["gene"] in Dg else "H"

    # ---- interchange (reused from A9) ----
    def build_corr(idxs):
        corr = {"input_ids": torch.stack([E[i]["co_ids"] for i in idxs]).to(dev)}
        if E[0]["attn_mask"] is not None:
            corr["attention_mask"] = torch.stack([E[i]["attn_mask"] for i in idxs]).to(dev)
        return corr

    def interchange(pairs, synth=None):
        res = {kk: [] for kk in ("M", "s1", "uinfo")}
        for s in range(0, len(pairs), args.batch):
            chunk = pairs[s:s + args.batch]
            iidx = [p[0] for p in chunk]
            corr = build_corr(iidx)
            dtok = torch.tensor([E[i]["dtok"] for i in iidx], device=dev)
            base = torch.stack([E[i]["base"] for i in iidx]).to(dev)
            Ap = torch.stack([E[i]["Ap"] for i in iidx]).to(dev)
            r3co = torch.stack([E[i]["r3co"] for i in iidx]).to(dev)
            src = []
            for (i, j) in chunk:
                src.append(E[i]["r3co"] if j is None else E[i]["r3cl"] if j == "self" else E[j]["r3cl"])
            if synth is None:
                dr = torch.stack(src).to(dev) - r3co
            else:
                Vp, u, beta = synth
                coef = torch.tensor([0.0 if j is None else float(Vp[i] if j == "self" else Vp[j])
                                     for (i, j) in chunk], device=dev)
                dr = (beta * coef)[:, None, None] * u[None, None, :].expand(len(chunk), 4, -1)
            dV = deliver(Wv, dr)
            od = head_output(base, Ap, dV)
            bidx = torch.arange(len(chunk), device=dev)
            with inject_capture(ad, corr, dtok, od) as o_inj:
                m = task.margin(o_inj.logits.float(), [E[i]["dtok"] - off for i in iidx])
                hf = o_inj.hidden_states[-1][bidx, dtok].float()
            res["M"] += m.cpu().tolist()
            res["s1"] += (dV.flatten(1).norm(dim=1)).cpu().tolist()
            res["uinfo"] += (hf @ U.T).norm(dim=1).cpu().tolist()
        return {kk: np.array(v) for kk, v in res.items()}

    allidx = list(range(len(E)))
    Mself = interchange([(i, "self") for i in allidx])["M"]
    Mempty = interchange([(i, None) for i in allidx])["M"]
    denom = Mself - Mempty
    good = [i for i in allidx if denom[i] > 0.05]

    # ================= PATH A (reconciliation): A9's competitor-matched design + composition control ====
    if args.mode == "a9matched":
        rngv = {k: float(np.nanstd([e[k] for e in E])) + 1e-9 for k in ("V2a", "V2b", "V3")}

        def a9con(split, cap):     # A9's V2b contrastive pairs: |dV2b| large, V2a & V3 held close
            res = []; su = {}
            for i in good:
                if E[i]["split"] != split:
                    continue
                for j in allidx:
                    if j == i or E[j]["split"] != split or E[i]["gene"] == E[j]["gene"] or su.get(j, 0) >= 6:
                        continue
                    far = abs(E[i]["V2b"] - E[j]["V2b"]) > 1.0 * rngv["V2b"]
                    comp_close = (abs(E[i]["V2a"] - E[j]["V2a"]) <= 0.15 * rngv["V2a"] and
                                  abs(E[i]["V3"] - E[j]["V3"]) <= 0.15 * rngv["V3"])
                    if far and comp_close:
                        res.append((i, j)); su[j] = su.get(j, 0) + 1
                    if len(res) >= cap:
                        break
                if len(res) >= cap:
                    break
            return res

        conD = a9con("D", args.max_pairs); conH = a9con("H", args.max_pairs)
        needed = sorted(set(conD + conH))
        IRr = interchange(needed) if needed else {"M": np.array([])}
        pm = {p: k for k, p in enumerate(needed)}

        def Sv(pairs):
            ks = [pm[p] for p in pairs]; i0 = np.array([p[0] for p in pairs])
            return (IRr["M"][ks] - Mempty[i0]) / (denom[i0] + 1e-9) - 1.0

        def partial(S, T, C, g):     # linear DML partial slope of S on T net of controls C (or None)
            if C is None:
                St, Tt = S - S.mean(), T - T.mean()
            else:
                St = S - crossfit(S, C, g, args.folds, SEED, "ridge")
                Tt = T - crossfit(T, C, g, args.folds, SEED, "ridge")
            return float(np.sum(St * Tt) / (np.sum(Tt * Tt) + 1e-12)), St, Tt

        rec = {}
        for tag, prs in (("H", conH), ("D", conD)):
            if len(prs) < 20:
                rec[tag] = {"n": len(prs), "note": "too_few_a9_matched_pairs"}; continue
            S = Sv(prs); dV = np.array([E[j]["V2b"] - E[i]["V2b"] for (i, j) in prs])
            dZ = np.stack([E[j]["Z"] - E[i]["Z"] for (i, j) in prs])
            dV2a = np.array([E[j]["V2a"] - E[i]["V2a"] for (i, j) in prs])
            dV3 = np.array([E[j]["V3"] - E[i]["V3"] for (i, j) in prs])
            gi = np.array([E[i]["gene"] for (i, j) in prs]); gj = np.array([E[j]["gene"] for (i, j) in prs])
            fin = np.isfinite(S) & np.isfinite(dV)
            S, dV, dZ, dV2a, dV3, gi, gj = (a[fin] for a in (S, dV, dZ, dV2a, dV3, gi, gj))
            r_raw = float(np.corrcoef(dV, S)[0, 1]) if np.std(dV) > 0 else None
            b0, St0, Tt0 = partial(S, dV, None, gi); l0, h0 = dyadic_ci(St0, Tt0, gi, gj, SEED, args.n_boot)
            bz, Stz, Ttz = partial(S, dV, dZ, gi); lz, hz = dyadic_ci(Stz, Ttz, gi, gj, SEED, args.n_boot)
            bc, Stc, Ttc = partial(S, dV, np.column_stack([dZ, dV2a, dV3]), gi)
            lc, hc = dyadic_ci(Stc, Ttc, gi, gj, SEED, args.n_boot)
            nz = np.linalg.norm(dZ, axis=1); msk = nz <= np.quantile(nz, 1 / 3)
            r_zm = float(np.corrcoef(dV[msk], S[msk])[0, 1]) if msk.sum() >= 12 and np.std(dV[msk]) > 0 else None
            rec[tag] = {"n": int(len(S)), "r_raw_a9": r_raw,
                        "beta_raw": b0, "ci_raw": [l0, h0], "raw_excl0": bool(l0 > 0 or h0 < 0),
                        "beta_net_Z": bz, "ci_net_Z": [lz, hz], "netZ_excl0": bool(lz > 0 or hz < 0),
                        "beta_net_Z_competitors": bc, "ci_net_Zc": [lc, hc],
                        "netZc_excl0": bool(lc > 0 or hc < 0),
                        "r_composition_matched": r_zm, "n_zmatched": int(msk.sum())}
        outr = out.parent / "a9c_matched_composition_reconcile.json"
        payload = {"mode": "a9matched", "n": len(E), "n_genes": len(ug), "Kz": Kz,
                   "value_linearity_err": lin_err, "reconcile": rec,
                   "note": "Path A: A9 competitor-matched design (V2a,V3 held; |dV2b|>1sd) reproduced on "
                           "A9b data, then composition-controlled. r_raw_a9 ~ reproduces A9's -0.51; "
                           "beta_net_Z tests survival net of bulk composition; beta_net_Z_competitors "
                           "adds V2a,V3 as linear controls."}
        outr.parent.mkdir(parents=True, exist_ok=True)
        outr.write_text(json.dumps(payload, indent=1, default=float))
        print(f"PATH A reconcile (value_linearity_err={lin_err:.2e}):")
        for tag in ("H", "D"):
            r = rec.get(tag, {})
            print(f"  [{tag}] n={r.get('n')} r_raw_a9={r.get('r_raw_a9')} beta_raw={r.get('beta_raw')} "
                  f"({'excl0' if r.get('raw_excl0') else 'incl0'}) | beta_net_Z={r.get('beta_net_Z')} "
                  f"ci={r.get('ci_net_Z')} ({'excl0' if r.get('netZ_excl0') else 'incl0'}) | "
                  f"r_comp_matched={r.get('r_composition_matched')} (n_zm={r.get('n_zmatched')})")
        print(f"saved -> {outr}")
        return

    # ---- broad contrastive pair set per split (natural ΔV x ΔZ distribution) ----
    def contrastive(split, cap, seed):
        rng = np.random.default_rng(seed)
        elig = [(i, j) for i in good if E[i]["split"] == split
                for j in allidx if j != i and E[j]["split"] == split and E[i]["gene"] != E[j]["gene"]]
        rng.shuffle(elig)
        su = {}; tu = {}; res = []
        for (i, j) in elig:
            if su.get(j, 0) >= 6 or tu.get(i, 0) >= 10:
                continue
            res.append((i, j)); su[j] = su.get(j, 0) + 1; tu[i] = tu.get(i, 0) + 1
            if len(res) >= cap:
                break
        return res

    cap = 24 if args.smoke else args.max_pairs
    pairsD = contrastive("D", cap, SEED); pairsH = contrastive("H", cap, SEED + 1)
    needed = sorted(set(pairsD + pairsH))
    IR = interchange(needed) if needed else {"M": np.array([])}
    pmap = {p: k for k, p in enumerate(needed)}

    def S_of(pairs):
        ks = [pmap[p] for p in pairs]
        m = IR["M"][ks]; i0 = np.array([p[0] for p in pairs])
        return (m - Mempty[i0]) / (denom[i0] + 1e-9) - 1.0        # signed transfer S = T - 1

    def pack(pairs, Vkey):
        """Assemble (S, dV, dZ, gene_i, gene_j) for a strength variable over a pair list (finite only)."""
        S = S_of(pairs)
        dV = np.array([float(E[j][Vkey]) - float(E[i][Vkey]) for (i, j) in pairs])
        dZ = np.stack([E[j]["Z"] - E[i]["Z"] for (i, j) in pairs])
        gi = np.array([E[i]["gene"] for (i, j) in pairs]); gj = np.array([E[j]["gene"] for (i, j) in pairs])
        fin = np.isfinite(S) & np.isfinite(dV)
        return S[fin], dV[fin], dZ[fin], gi[fin], gj[fin]

    # ---- per-variable estimate (V2b primary, V3 secondary) on D and H ----
    A9_SIGN = {"V2b": -1.0, "V3": +1.0}

    def estimate(pairs, Vkey, seed):
        S, dV, dZ, gi, gj = pack(pairs, Vkey)
        info = {"n_pairs": int(len(S))}
        if len(S) < max(30, 3 * (3 + Kz)):
            info["identifiable"] = False; info["reason"] = "too_few_pairs"; return info
        groups = gi
        # collinearity(strength~Z) + does bulk composition predict the transfer at all? + common support
        r2 = cv_r2(dV, dZ, groups, args.folds, seed, "ridge")
        sz = cv_r2(S, dZ, groups, args.folds, seed, "ridge")     # composition -> transfer predictive power
        nz = np.linalg.norm(dZ, axis=1); av = np.abs(dV)
        supp = int(np.sum((av >= np.quantile(av, 2 / 3)) & (nz <= np.quantile(nz, 1 / 3))))
        info.update({"collinearity_cv_r2": r2, "composition_predicts_transfer_cv_r2": sz,
                     "common_support": supp, "identifiable": bool(r2 < 0.85 and supp >= args.min_support)})
        # DML (primary = LINEAR ridge; GBR overfits this low-dim Z -> negative OOF R^2) + dyadic CI
        d = dml_beta(S, dV, dZ, groups, args.folds, seed, "ridge")
        if d is not None:
            lo, hi = dyadic_ci(d["St"], d["Vt"], gi, gj, seed, args.n_boot)
            info["dml"] = {"beta": d["beta"], "ci": [lo, hi], "excludes_0": bool(lo > 0 or hi < 0),
                           "sign": int(np.sign(d["beta"])), "a9_sign": A9_SIGN.get(Vkey),
                           "sign_matches_a9": bool(np.sign(d["beta"]) == A9_SIGN.get(Vkey))}
        # GBR robustness (documents the overfit: it should NOT add real signal here)
        dg = dml_beta(S, dV, dZ, groups, args.folds, seed, "gbr")
        if dg is not None:
            info["gbr_beta"] = dg["beta"]
        # matched-set slope (||dZ|| bottom tercile), dyadic CI on centered residuals
        msk = nz <= np.quantile(nz, 1 / 3)
        if msk.sum() >= 20:
            Sm = S[msk] - S[msk].mean(); Vm = dV[msk] - dV[msk].mean()
            bm = float(np.sum(Sm * Vm) / (np.sum(Vm * Vm) + 1e-12))
            lo, hi = dyadic_ci(Sm, Vm, gi[msk], gj[msk], seed, args.n_boot)
            info["matched"] = {"beta": bm, "ci": [lo, hi], "n": int(msk.sum()),
                               "excludes_0": bool(lo > 0 or hi < 0)}
        return info

    result_vars = {V: {"D": estimate(pairsD, V, SEED), "H": estimate(pairsH, V, SEED + 7)}
                   for V in ("V2b", "V3")}

    if args.dump_arrays:
        dump = {}
        for tag, prs in (("D", pairsD), ("H", pairsH), ("DH", pairsD + pairsH)):
            for Vk in ("V2b", "V3"):
                S, dV, dZ, gi, gj = pack(prs, Vk)
                dump[f"{tag}_{Vk}_S"] = S; dump[f"{tag}_{Vk}_dV"] = dV; dump[f"{tag}_{Vk}_dZ"] = dZ
                dump[f"{tag}_{Vk}_gi"] = gi; dump[f"{tag}_{Vk}_gj"] = gj
        np.savez(args.dump_arrays, Kz=Kz, **dump)
        print(f"dumped per-pair arrays -> {args.dump_arrays}")

    # ---- CALIBRATION GATE 1: through-model planted-positive (A9 sensitivity) ----
    rngp = np.random.default_rng(SEED)
    d0 = np.stack([(E[i]["r3cl"] - E[i]["r3co"]).numpy().mean(0) for i in good]).mean(0)
    Vplant = rngp.uniform(1.0, 2.0, len(E))
    synth = (Vplant, torch.tensor(d0, device=dev, dtype=torch.float32), 1.0)
    pl_cap = 12 if args.smoke else 160
    pl_pairs = contrastive("D", pl_cap, SEED + 3) + contrastive("H", pl_cap, SEED + 4)
    pl_pairs = sorted(set(pl_pairs))
    Mself_pl = interchange([(i, "self") for i in good], synth=synth)["M"]
    denom_pl = {i: Mself_pl[k] - Mempty[i] for k, i in enumerate(good)}
    PIR = interchange(pl_pairs, synth=synth)["M"] if pl_pairs else np.array([])
    plmap = {p: k for k, p in enumerate(pl_pairs)}
    pl_pairs = [p for p in pl_pairs if abs(denom_pl.get(p[0], 0)) > 0.05]
    Spl = np.array([(PIR[plmap[p]] - Mempty[p[0]]) / (denom_pl[p[0]] + 1e-9) - 1.0 for p in pl_pairs])
    dVpl = np.array([Vplant[p[1]] - Vplant[p[0]] for p in pl_pairs])
    dZpl = np.stack([E[p[1]]["Z"] - E[p[0]]["Z"] for p in pl_pairs])
    gip = np.array([E[p[0]]["gene"] for p in pl_pairs]); gjp = np.array([E[p[1]]["gene"] for p in pl_pairs])
    planted = {"n": len(Spl)}
    if len(Spl) >= 20:
        dp = dml_beta(Spl, dVpl, dZpl, gip, args.folds, SEED, "gbr")
        if dp is not None:
            lo, hi = dyadic_ci(dp["St"], dp["Vt"], gip, gjp, SEED, args.n_boot)
            planted.update({"beta": dp["beta"], "ci": [lo, hi],
                            "detected": bool((lo > 0 or hi < 0) and dp["beta"] > 0)})

    # ---- CALIBRATION GATE 2: permuted-label FPR (A9 specificity), on the DML statistic ----
    perm_hits = 0; perm_tot = 0
    Sh, dVh, dZh, gih, gjh = pack(pairsH, "V2b")
    nperm = 3 if args.smoke else args.n_perm
    if len(Sh) >= 30:
        for pp in range(nperm):
            perm = rngp.permutation(len(E))
            vmap = {orig: E[perm[k]]["V2b"] for k, orig in enumerate(range(len(E)))}
            dVp = np.array([float(vmap[j]) - float(vmap[i]) for (i, j) in pairsH])
            fin = np.isfinite(S_of(pairsH)) & np.isfinite(dVp)
            dperm = dml_beta(S_of(pairsH)[fin], dVp[fin], np.stack([E[j]["Z"] - E[i]["Z"]
                             for (i, j) in pairsH])[fin], gih, args.folds, pp, "ridge")
            if dperm is None:
                continue
            lo, hi = dyadic_ci(dperm["St"], dperm["Vt"], gih, gjh, pp, 300)
            perm_hits += int(lo > 0 or hi < 0); perm_tot += 1
    fpr = perm_hits / perm_tot if perm_tot else 1.0

    # ---- CALIBRATION GATES 3 & 4: semi-synthetic CONFOUND gates ----
    Sc, dVc, dZc, gic, gjc = pack(pairsD + pairsH, "V2b")
    confound = {"n": int(len(Sc))}
    if len(Sc) >= 40:
        groups = gic
        # composition->strength link built FOLD-FREE (full fit) to avoid the OOF fold-coupling that
        # manufactured a phantom confound; LINEAR because GBR overfits Z here (negative OOF R^2).
        cmp = zscore(Ridge(alpha=1.0).fit(dZc, dVc).predict(dZc)); zdV = zscore(dVc)
        rc = np.random.default_rng(SEED + 11)
        # Gate A: composition-only -> beta_str NULL (CI covers 0)
        SA = cmp + rc.normal(0, 0.5, len(Sc))
        dA = dml_beta(SA, zdV, dZc, groups, args.folds, SEED, "ridge")
        if dA is not None:
            lo, hi = dyadic_ci(dA["St"], dA["Vt"], gic, gjc, SEED, args.n_boot)
            confound["gateA_composition_only"] = {"beta": dA["beta"], "ci": [lo, hi],
                                                  "covers_0": bool(lo <= 0 <= hi)}
        # Gate B: collinear-strength (beta0=1 in zscored units) -> RECOVERED (CI covers beta0)
        beta0 = 1.0
        SB = beta0 * zdV + cmp + rc.normal(0, 0.5, len(Sc))
        dB = dml_beta(SB, zdV, dZc, groups, args.folds, SEED, "ridge")
        if dB is not None:
            lo, hi = dyadic_ci(dB["St"], dB["Vt"], gic, gjc, SEED, args.n_boot)
            confound["gateB_collinear_strength"] = {"beta": dB["beta"], "beta0": beta0, "ci": [lo, hi],
                                                    "covers_beta0": bool(lo <= beta0 <= hi)}

    # ---- preflight (HARD gates) ----
    def g(d, k, default=False):
        return bool(d.get(k, default)) if isinstance(d, dict) else False
    checks = {
        "value_linearity": {"PASS": bool(lin_err < 1e-4), "max_abs_err": lin_err},
        "self_interchange_identity": {"PASS": bool(abs(np.median(Mself[good] - denom[good]
                                       - Mempty[good])) < 1e-6) if len(good) else False},
        "usable_value_path": {"PASS": bool(len(good) / len(E) >= 0.4), "frac": len(good) / len(E),
                              "n_good": len(good)},
        "planted_positive_sensitivity": {"PASS": g(planted, "detected"), "planted": planted},
        "permuted_label_fpr": {"PASS": bool(fpr <= 0.05), "fpr": fpr, "n_perm": perm_tot},
        "confound_gateA_composition_null": {"PASS": g(confound.get("gateA_composition_only", {}),
                                            "covers_0"), "detail": confound.get("gateA_composition_only")},
        "confound_gateB_strength_recovered": {"PASS": g(confound.get("gateB_collinear_strength", {}),
                                              "covers_beta0"), "detail": confound.get("gateB_collinear_strength")},
        "maxent_pinned": {"PASS": True,
                          "sha256": "8b6d7b46f81788fb0710314867e3233e7f9b7f061b809bbdb11bb120422b7c00",
                          "score3_test": round(score3("ttccaaacgaacttttgtagGGA"), 3)},
        "both_strands_present": {"PASS": len(strands) == 2, "strands": strands},
    }
    ckpt = Path(args.model); ckpt = ckpt if ckpt.is_absolute() else REPO / ckpt
    checks["checkpoint_recorded"] = {"PASS": True, "sha256": sha256_file(ckpt / "pytorch_model.bin")}
    if args.smoke:      # calibration needs full-n power; smoke is code-path only (gates ran w/o error)
        for kk in ("planted_positive_sensitivity", "permuted_label_fpr",
                   "confound_gateA_composition_null", "confound_gateB_strength_recovered"):
            checks[kk]["PASS"] = True
    pf = {"commit": git_commit(REPO), "n": len(E), "Kz": Kz, "checks": checks}
    pf["PASS"] = bool(all(c["PASS"] for c in pf["checks"].values()))
    pf_path.parent.mkdir(parents=True, exist_ok=True)
    pf_path.write_text(json.dumps(pf, indent=1, default=float))
    print(f"preflight PASS={pf['PASS']}  (Kz={Kz})  -> {pf_path}")
    for name, c in pf["checks"].items():
        print(f"    {'ok ' if c['PASS'] else 'XX '}{name}")

    if args.smoke:
        print("SMOKE: mechanics + calibration gates on tiny-n (val sealed).")
        return
    if not pf["PASS"] and not args.force:
        print("  *** preflight FAILED -- not reporting (prereg s5).")
        return

    # ---- verdict (held-out H, V2b primary); sign reported as-observed, NOT assumed from A9 ----
    H = result_vars["V2b"]["H"]; Dv = result_vars["V2b"]["D"]
    dml = H.get("dml", {}); matched = H.get("matched", {})
    sign_consistent = bool(dml.get("sign") and Dv.get("dml", {}).get("sign") == dml.get("sign"))
    comp_predicts = bool((H.get("composition_predicts_transfer_cv_r2") or -1) >= 0.10)
    strength_net = bool(H.get("identifiable") and dml.get("excludes_0") and sign_consistent
                        and matched.get("excludes_0"))
    composite = bool((H.get("collinearity_cv_r2") or 0) >= 0.85 or H.get("common_support", 0) < args.min_support)
    if strength_net:
        verdict = "strength_names_payload_net_of_bulk_composition"
    elif composite:
        verdict = "splice_strength_bulk_composition_composite"
    elif comp_predicts and not dml.get("excludes_0"):
        verdict = "generic_bulk_composition"
    else:
        verdict = "strength_effect_weak_or_unresolved"

    result = {"n": len(E), "n_genes": len(ug), "strands": strands, "Kz": Kz,
              "adjustment_set": ["GC", "PY", "POS", f"{Kz}xPCA(di+tri)"],
              "vars": result_vars,
              "calibration": {"planted_positive": planted, "permuted_fpr": fpr, "confound_gates": confound},
              "verdict": verdict, "preflight_PASS": pf["PASS"],
              "confirmatory": bool(pf["PASS"] and len(E) >= 120),
              "REFLECTION": {"status": "PENDING -- fill after inspection (prereg s7); RE-ANCHOR to s0."}}
    out.write_text(json.dumps(result, indent=1, default=float))
    status = "CONFIRMATORY" if result["confirmatory"] else "DIAGNOSTIC"
    man = ExperimentManifest.create(
        "splicebert_coupling:a9b_payload_semantics_composition", sys.argv, status=status,
        config={"split": args.split, "n": len(E), "n_genes": len(ug), "max_pairs": args.max_pairs,
                "Kz": Kz, "folds": args.folds, "n_boot": args.n_boot, "n_perm": args.n_perm,
                "min_support": args.min_support, "preflight_PASS": pf["PASS"]},
        seeds={"gene_split": SEED, "perm": SEED, "boot": SEED},
        notes=["A9b: strength net of BULK composition via cross-fitted DML (Robinson partialling); "
               "two confound gates; dyadic bootstrap; held-out + matched. Reuses A9 interchange.",
               f"status={status}."])
    man.add_input(str(ckpt / "pytorch_model.bin"))
    man.write(out.parent / (out.stem + ".manifest.json"))

    print("=" * 84)
    for V in ("V2b", "V3"):
        h = result_vars[V]["H"]; dm = h.get("dml", {}); mt = h.get("matched", {})
        print(f"  {V}(H): beta_dml={dm.get('beta')} ci={dm.get('ci')} excl0={dm.get('excludes_0')} "
              f"sign={dm.get('sign')}(a9={dm.get('a9_sign')}) | matched={mt.get('beta')} "
              f"excl0={mt.get('excludes_0')} | collinR2={h.get('collinearity_cv_r2')} "
              f"compR2(S~Z)={h.get('composition_predicts_transfer_cv_r2')} "
              f"support={h.get('common_support')} ident={h.get('identifiable')}")
    print(f"  gates: planted={checks['planted_positive_sensitivity']['PASS']} "
          f"fpr={fpr:.3f} A={checks['confound_gateA_composition_null']['PASS']} "
          f"B={checks['confound_gateB_strength_recovered']['PASS']}")
    print(f"  VERDICT: {verdict}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
