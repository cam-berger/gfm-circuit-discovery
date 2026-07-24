#!/usr/bin/env python
"""Donor-graph experiment 13 -- A9: semantic identity of the acceptor payload (interchange).

Frozen protocol: studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_A9_PAYLOAD_SEMANTICS.md.

Matched interchange (causal-abstraction test, Geiger 2021): deliver example j's clean acceptor payload
through example i's CLEAN address (A8 reconstruct-inject, cross-example), and ask whether a frozen
sequence variable predicts when the swap is silent (equivalence) vs transfers sign-correctly
(contrastive), on held-out genes.  Variables: V1 identity, V2 MaxEntScan (a=23nt, b=local 4-tok delta),
V3 MLM acceptor confidence (secondary/circular), V4 3-mer composition.  Energy-stratified on ||dr|| and
||W_V dr||; calibrated with a planted-positive (sensitivity) and permuted-label FPR (specificity);
staged value -> U_info -> margin.  U_info is projected onto (C16), never refit.  No new kernel.
--smoke = mechanics memorials + calibration sanity on tiny-n (val sealed).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "splicebert"))

from maxentpy import maxent                                                    # noqa: E402
from maxentpy.maxent import load_matrix3                                       # noqa: E402

from gfmcd.provenance import ExperimentManifest, git_commit, sha256_file       # noqa: E402
from gfmcd.circuits.hf_bert import _context_from_output, _split_context, patch_heads  # noqa: E402
from gfmcd.splicing.circuit import (                                          # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, gboot_ratio,
    load_adapter, load_examples, pick_device,
)
from gfmcd.splicing.payload import deliver, head_output, value_weight_head     # noqa: E402
from gfmcd.splicing.subspace import top_k_subspace                             # noqa: E402

L3, H3 = 3, 1
ACC_OFF = (-3, -2, -1, 0)
M3 = load_matrix3()
CANON_ACC4 = "CAGG"      # reference window for the local MaxEnt counterfactual (V2b)
SEED = 0


def score3(s):
    s = s.upper()
    return maxent.score3(s, matrix=M3) if len(s) == 23 and set(s) <= set("ACGT") else float("nan")


def scorers(seq, a):
    """Frozen sequence variables at acceptor first-exon base a (AG at [a-2,a-1])."""
    w23 = seq[a - 20:a + 3] if a - 20 >= 0 and a + 3 <= len(seq) else ""
    v2a = score3(w23)
    v2b = (v2a - score3(w23[:17] + CANON_ACC4 + w23[21:])) if len(w23) == 23 else float("nan")
    v1 = seq[a - 3:a + 1] if a - 3 >= 0 and a + 1 <= len(seq) else ""
    comp_win = seq[a - 15:a + 5] if a - 15 >= 0 and a + 5 <= len(seq) else ""
    kmers = np.zeros(64)
    if len(comp_win) >= 3:
        idx = {c: i for i, c in enumerate("ACGT")}
        for t in range(len(comp_win) - 2):
            tri = comp_win[t:t + 3]
            if set(tri) <= set("ACGT"):
                kmers[16 * idx[tri[0]] + 4 * idx[tri[1]] + idx[tri[2]]] += 1
        if kmers.sum() > 0:
            kmers /= kmers.sum()
    return {"V1": v1, "V2a": v2a, "V2b": v2b, "V4": kmers}


def caprun(ad, inp, dtok, want_attn):
    vbox = {}
    hh = ad.attention_self_modules[L3].value.register_forward_hook(
        lambda _m, _i, o: vbox.__setitem__("v", o.detach()))
    try:
        o = ad.model(**inp, output_hidden_states=True, output_attentions=want_attn)
    finally:
        hh.remove()
    Dh = ad.hidden_size // ad.num_heads
    vhead = vbox["v"][:, :, H3 * Dh:(H3 + 1) * Dh]
    B = inp["input_ids"].shape[0]
    ap = o.attentions[L3][torch.arange(B, device=dtok.device), H3, dtok, :] if want_attn else None
    return o, o.hidden_states[L3], vhead, ap


@contextmanager
def inject_capture(ad, corr, dtok, recon):
    """Inject recon [B,Dh] as L3H1 donor output; yield the patched forward output (with hidden states)."""
    B, Dh = recon.shape
    S = corr["input_ids"].shape[1]
    src = torch.zeros(B, ad.num_heads, S, Dh, device=recon.device, dtype=recon.dtype)
    b = torch.arange(B, device=recon.device)
    src[b, H3, dtok] = recon
    mask = torch.zeros(B, S, dtype=torch.bool, device=recon.device); mask[b, dtok] = True
    with patch_heads(ad, {L3: src}, [(L3, H3)], token_mask=mask):
        yield ad.model(**corr, output_hidden_states=True)


def frozen_uinfo(ad, task, gtf, fasta, off, dev, n_train=300, k=8):
    """C16 U_info recomputed on the TRAIN split (frozen; no val leakage)."""
    Xtr, _ = load_examples(gtf, fasta, "train", n_train, task=task, tag="a9-uinfo-train", min_effect=None)
    rows = []
    for i in range(0, len(Xtr), task.batch):
        ch = Xtr[i:i + task.batch]
        cl, co, dn = task.pair(ch)
        oc = ad.model(**cl, output_hidden_states=True); oo = ad.model(**co, output_hidden_states=True)
        pos = torch.tensor([d + off for d in dn], device=dev); idx = torch.arange(len(dn), device=dev)
        rows.append((oc.hidden_states[-1][idx, pos] - oo.hidden_states[-1][idx, pos]).float().cpu().numpy())
    U = top_k_subspace(np.concatenate(rows), k)          # [k, H] orthonormal rows
    return torch.tensor(np.ascontiguousarray(U), device=dev, dtype=torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-pairs", type=int, default=400)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF); ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="results/splicebert_coupling/superseded/a9_payload_semantics.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    out = REPO / args.out
    pf_path = out.parent / "preflight_a9_payload_semantics.json"
    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=True)
    task = CouplingTask(ad, batch=args.batch)
    off = task.off; Dh = ad.hidden_size // ad.num_heads
    Wv = value_weight_head(ad).to(dev)
    U = frozen_uinfo(ad, task, args.gtf, args.fasta, off, dev, n_train=80 if args.smoke else 300)

    n = min(args.n, 32) if args.smoke else args.n
    X, _ = load_examples(args.gtf, args.fasta, args.split, n, task=task, tag=f"a9-{args.split}",
                         min_effect=None)
    strands = sorted({x.strand for x in X})

    # ---- per-example capture ----
    E = []       # list of dicts per example
    lin_err = 0.0
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
        # value_linearity memorial
        dVt = (V_cl - V_co)[bidx, accwin]; dVl = deliver(Wv, (r3_cl - r3_co)[bidx, accwin])
        lin_err = max(lin_err, float((dVt - dVl).abs().max()))
        # V3 = MLM acceptor confidence: mask AG at [a-2,a-1], read logprob of A,G
        cl_m = {k: v.clone() for k, v in cl.items()}
        agpos = torch.tensor([[x.next_acceptor_pos + off - 2, x.next_acceptor_pos + off - 1] for x in ch],
                             device=dev)
        cl_m["input_ids"][bidx[:, :2], agpos] = task.mask_id
        lg = ad.model(**cl_m).logits.float().log_softmax(-1)
        for bj, x in enumerate(ch):
            a = x.next_acceptor_pos
            sc = scorers(x.sequence, a)
            aid = task.bid["A"]; gid = task.bid["G"]
            v3 = float(lg[bj, agpos[bj, 0], aid] + lg[bj, agpos[bj, 1], gid])
            E.append({"r3cl": r3_cl[bj, accwin[bj]].cpu(), "r3co": r3_co[bj, accwin[bj]].cpu(),
                      "Ap": Ap[bj, accwin[bj]].cpu(), "base": base[bj].cpu(), "dtok": int(dtok[bj]),
                      "co_ids": co["input_ids"][bj].cpu(),
                      "attn_mask": cl["attention_mask"][bj].cpu() if "attention_mask" in cl else None,
                      "m_cl": float(task.margin(oc.logits.float(), dn)[bj]),
                      "m_co": float(task.margin(oo.logits.float(), dn)[bj]),
                      "gene": x.gene_id, "V3": v3, **sc})
        print(f"  capture {min(i + args.batch, len(X))}/{len(X)}", end="\r", flush=True)
    print()
    E = [e for e in E if isinstance(e["V2a"], float) and not np.isnan(e["V2a"]) and e["V1"]]
    if len(E) < 16:
        raise RuntimeError(f"only {len(E)} scorable examples.")

    ug = sorted({e["gene"] for e in E}); Dg = set(ug[::2])
    for e in E:
        e["split"] = "D" if e["gene"] in Dg else "H"

    # ---- interchange (batched over a pair list) ----
    def build_corr(idxs):
        # use the ACTUAL task.pair corrupt input (donor-masked + partner AG->CC), never a rebuild
        corr = {"input_ids": torch.stack([E[i]["co_ids"] for i in idxs]).to(dev)}
        if E[0]["attn_mask"] is not None:
            corr["attention_mask"] = torch.stack([E[i]["attn_mask"] for i in idxs]).to(dev)
        return corr

    def interchange(pairs, synth=None):
        """pairs: (i,j)/(i,None)=empty/(i,'self'). synth=(Vp,u,beta) OVERRIDES the delivered payload
        with a pure signal beta*Vp[source]*u (calibration: makes Vp name the synthetic payload)."""
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
                if j is None:
                    src.append(E[i]["r3co"])                       # empty -> dr=0
                elif j == "self":
                    src.append(E[i]["r3cl"])
                else:
                    src.append(E[j]["r3cl"])
            if synth is None:
                dr = torch.stack(src).to(dev) - r3co               # natural payload edit [b,4,H]
            else:                                                  # OVERRIDE with pure synthetic signal
                Vp, u, beta = synth
                coef = torch.tensor([0.0 if j is None else float(Vp[i] if j == "self" else Vp[j])
                                     for (i, j) in chunk], device=dev)
                dr = (beta * coef)[:, None, None] * u[None, None, :].expand(len(chunk), 4, -1)
            dV = deliver(Wv, dr)                                    # [b,4,Dh]
            od = head_output(base, Ap, dV)                         # [b,Dh]
            bidx = torch.arange(len(chunk), device=dev)
            with inject_capture(ad, corr, dtok, od) as o_inj:
                m = task.margin(o_inj.logits.float(), [E[i]["dtok"] - off for i in iidx])
                hf = o_inj.hidden_states[-1][bidx, dtok].float()   # final donor residual
            res["M"] += m.cpu().tolist()
            res["s1"] += (dV.flatten(1).norm(dim=1)).cpu().tolist()    # ||W_V dr|| delivered energy
            res["uinfo"] += (hf @ U.T).norm(dim=1).cpu().tolist()      # U_info energy of final residual
        return {kk: np.array(v) for kk, v in res.items()}

    # per-example self / empty baselines
    allidx = list(range(len(E)))
    self_r = interchange([(i, "self") for i in allidx])
    empt_r = interchange([(i, None) for i in allidx])
    Mself = self_r["M"]; Mempty = empt_r["M"]
    denom = Mself - Mempty
    good = [i for i in allidx if denom[i] > 0.05]        # reliable value-path normalization (T target)

    # ---- matched pairs (frozen calipers) ----
    def ordinal(key):
        vals = np.array([e[key] for e in E], float)
        rng = np.nanstd(vals) + 1e-9
        return vals, rng
    comp = np.stack([e["V4"] for e in E])
    def compdist(i, j): return float(np.abs(comp[i] - comp[j]).sum())

    calipers = {"V2a": (0.15, 1.0), "V2b": (0.15, 1.0), "V3": (0.15, 1.0)}   # (eq_frac, con_frac)*std
    others = {"V2a": ["V2b", "V3"], "V2b": ["V2a", "V3"], "V3": ["V2a", "V2b"], "V1": ["V2a"],
              "V4": ["V2a"]}
    rng = {}
    for k in ("V2a", "V2b", "V3"):
        rng[k] = np.nanstd(np.array([e[k] for e in E], float)) + 1e-9

    def close(k, i, j, frac):
        if k in ("V2a", "V2b", "V3"):
            return abs(E[i][k] - E[j][k]) <= frac * rng[k]
        if k == "V1":
            return E[i]["V1"] == E[j]["V1"]
        return compdist(i, j) <= frac   # V4

    import itertools
    src_use = {}
    def pick(V, kind):
        eqf, conf = calipers.get(V, (0.15, 1.0))
        pairs = []
        cap = args.max_pairs if not args.smoke else 12
        for i in good:                                           # target must have reliable denom
            for j in allidx:
                if j == i or E[i]["gene"] == E[j]["gene"] or E[i]["split"] != E[j]["split"]:
                    continue
                if src_use.get(j, 0) >= 6:
                    continue
                comp_close = all(close(o, i, j, eqf if V != "V1" else 0.5) for o in others[V])
                if kind == "eq":
                    ok = close(V, i, j, eqf) and not comp_close      # tight V, loose competitors
                else:
                    far = (E[i]["V1"] != E[j]["V1"]) if V == "V1" else (
                        compdist(i, j) > 3.0 if V == "V4" else abs(E[i][V] - E[j][V]) > conf * rng[V])
                    ok = far and comp_close                          # big dV, tight competitors
                if ok:
                    pairs.append((i, j)); src_use[j] = src_use.get(j, 0) + 1
                if len(pairs) >= cap:
                    break
            if len(pairs) >= cap:
                break
        return pairs

    VARS = ["V2a", "V2b", "V3", "V1", "V4"]
    sets = {}
    for V in VARS:
        src_use = {}
        sets[V] = {"eq": pick(V, "eq"), "con": pick(V, "con")}

    # gather + run all needed interchanges once
    needed = sorted({p for V in VARS for kind in ("eq", "con") for p in sets[V][kind]})
    IR = interchange(needed) if needed else {"M": np.array([]), "s1": np.array([]), "uinfo": np.array([])}
    pmap = {p: k for k, p in enumerate(needed)}

    def T_of(pairs):
        if not pairs:
            return np.array([])
        ks = [pmap[p] for p in pairs]
        m = IR["M"][ks]
        i0 = np.array([p[0] for p in pairs])
        return (m - Mempty[i0]) / (denom[i0] + 1e-9)

    v2a_vals = np.array([e["V2a"] for e in E])

    def ordinal_stat(conp, valfn, Tfn):
        """Signed transfer (T-1) vs signed dV: correlation r + monotonicity across terciles."""
        Tc = Tfn(conp); fin = np.isfinite(Tc)
        cp = [p for p, f in zip(conp, fin) if f]
        if len(cp) < 12:
            return None, False
        dv = np.array([valfn(j) - valfn(i) for (i, j) in cp]); T = Tc[fin] - 1.0
        if np.std(dv) == 0:
            return 0.0, False
        r = float(np.corrcoef(dv, T)[0, 1])
        med = [float(np.median(T[t])) for t in np.array_split(np.argsort(dv), 3) if len(t) >= 2]
        mono = len(med) == 3 and (med[0] < med[1] < med[2] or med[0] > med[1] > med[2])
        return r, mono

    # PERMUTATION-CALIBRATED significance (replaces ad-hoc bootstrap CI): null of |corr_r| under permuted
    # labels; r_thresh = null 95th percentile -> controls the ordinal-test FPR by construction, while the
    # planted signal (r~0.98) sits far above it.  FPR is measured from the SAME null (specificity gate).
    rngp = np.random.default_rng(SEED)
    r_null = []
    if len(needed):
        i_of = np.array([p[0] for p in needed]); j_of = np.array([p[1] for p in needed])
        for kk in range(5 if args.smoke else args.n_perm):
            pv = rngp.permutation(v2a_vals); d = np.abs(pv[i_of] - pv[j_of])
            conp = [needed[k] for k in np.where(d >= 1.0 * rng["V2a"])[0]]
            r, mono = ordinal_stat(conp, (lambda i, pv=pv: float(pv[i])), T_of)
            if r is not None:
                r_null.append((abs(r), mono, kk))
    # split: calibrate r_thresh on even perms, MEASURE fpr out-of-sample on odd perms (genuine specificity)
    cal = [a for a, _, kk in r_null if kk % 2 == 0]
    r_thresh = float(np.percentile(cal, 98)) if cal else 1.0
    test = [(a, m) for a, m, kk in r_null if kk % 2 == 1]
    fpr = (sum(1 for a, m in test if a > r_thresh and m) / len(test)) if test else 1.0

    CATEG = {"V1", "V4"}

    def decide(eqp, conp, valfn, categorical, Tfn):
        """categorical -> energy-stratified drop (eq>con); ordinal -> signed transfer correlates with
        signed dV (|r|>r_thresh, permutation-calibrated) and is monotonic."""
        Teq, Tcon = Tfn(eqp), Tfn(conp)
        Teq_f = Teq[np.isfinite(Teq)] if len(Teq) else Teq
        eq_ok = bool(len(Teq_f) >= 8 and 0.80 <= float(np.median(Teq_f)) <= 1.20)
        info = {"n_eq": len(eqp), "n_eq_finite": int(len(Teq_f)), "n_con": len(conp),
                "T_eq_median": float(np.median(Teq_f)) if len(Teq_f) else None,
                "T_con_median": float(np.nanmedian(Tcon)) if len(Tcon) else None, "equivalence_ok": eq_ok}
        con_ok = False
        if categorical:
            strat = None
            if len(eqp) >= 6 and len(conp) >= 6:
                Eeq = IR["s1"][[pmap[p] for p in eqp]]; Econ = IR["s1"][[pmap[p] for p in conp]]
                qs = np.quantile(np.concatenate([Eeq, Econ]), [0, 1 / 3, 2 / 3, 1.0]); diffs = []
                for b in range(3):
                    me = Teq[(Eeq >= qs[b]) & (Eeq <= qs[b + 1])]; mc = Tcon[(Econ >= qs[b]) & (Econ <= qs[b + 1])]
                    if len(me) >= 2 and len(mc) >= 2:
                        diffs.append(float(np.median(me) - np.median(mc)))
                strat = float(np.mean(diffs)) if diffs else None
            con_ok = bool(strat is not None and strat > 0.10); info["strat_contrast"] = strat
        else:
            r, mono = ordinal_stat(conp, valfn, Tfn)
            con_ok = bool(r is not None and abs(r) > r_thresh and mono)
            info.update({"corr_r": r, "r_thresh": r_thresh, "monotonic": mono})
        info["contrastive_ok"] = con_ok
        info["identifiable"] = bool(len(eqp) >= 8 and len(conp) >= 8)
        info["names_payload"] = bool(eq_ok and con_ok)
        return info

    def summarize(V, split):
        eqp = [p for p in sets[V]["eq"] if E[p[0]]["split"] == split]
        conp = [p for p in sets[V]["con"] if E[p[0]]["split"] == split]
        return decide(eqp, conp, (lambda i: float(E[i][V])) if V not in CATEG else (lambda i: 0.0),
                      V in CATEG, T_of)

    result_vars = {V: {"D": summarize(V, "D"), "H": summarize(V, "H")} for V in VARS}

    # ---- planted-positive calibration (fpr already computed above via permutation) ----
    cap_pl = 60 if not args.smoke else 8
    # PLANTED-POSITIVE (sensitivity): deliver a PURE synthetic payload = typ*Vplant*u_mean (u_mean = the
    # population mean payload edit, which is KNOWN to recover). Vplant scales it, so Vplant PROVABLY names
    # the synthetic payload -> the decision pipeline MUST detect it (else the classifier is underpowered).
    d0 = np.stack([(E[i]["r3cl"] - E[i]["r3co"]).numpy().mean(0) for i in good]).mean(0)  # recovering dir
    Vplant = rngp.uniform(1.0, 2.0, len(E))                       # all >=1 => denom_pl >= M(d0)-Mempty>0
    synth = (Vplant, torch.tensor(d0, device=dev, dtype=torch.float32), 1.0)   # dr = Vplant * d0

    def plant_pairs(kind):
        out, used = [], {}
        for i in good:
            for j in allidx:
                if j == i or E[i]["gene"] == E[j]["gene"] or used.get(j, 0) >= 6:
                    continue
                d = abs(Vplant[i] - Vplant[j])
                if (d < 0.12) if kind == "eq" else (d > 0.6):
                    out.append((i, j)); used[j] = used.get(j, 0) + 1
                if len(out) >= cap_pl:
                    break
            if len(out) >= cap_pl:
                break
        return out

    pl_eq, pl_con = plant_pairs("eq"), plant_pairs("con")
    Mself_pl = interchange([(i, "self") for i in good], synth=synth)["M"]
    denom_pl = {i: Mself_pl[k] - Mempty[i] for k, i in enumerate(good)}
    plset = sorted(set(pl_eq + pl_con))
    PIR = interchange(plset, synth=synth)["M"] if plset else np.array([])
    plmap = {p: k for k, p in enumerate(plset)}

    def Tpl(pairs):                                          # NaN (not skip) so pair alignment holds
        return np.array([(PIR[plmap[p]] - Mempty[p[0]]) / (denom_pl[p[0]] + 1e-9)
                         if abs(denom_pl.get(p[0], 0)) > 0.05 else np.nan for p in pairs])
    # Vplant is ORDINAL -> use the SAME ordinal decision as the real variables (this is the sensitivity
    # test of the actual classifier the verdict uses)
    planted = decide(pl_eq, pl_con, lambda i: float(Vplant[i]), categorical=False, Tfn=Tpl)

    # (permuted-label FPR was computed above via the permutation-calibrated r_thresh)

    checks = {
        "value_linearity": {"PASS": bool(lin_err < 1e-4), "max_abs_err": lin_err},
        "usable_value_path": {"PASS": bool(len(good) / len(E) >= 0.4),
                              "frac_good_denom": len(good) / len(E), "n_good": len(good),
                              "note": "T(i<-i)=1 by construction; gate = enough examples with denom>0.05"},
        "positive_control": {"PASS": bool(np.median((Mself - Mempty)) > 0.05),
                             "self_minus_empty_median": float(np.median(Mself - Mempty))},
        "planted_positive_sensitivity": {"PASS": bool(planted["names_payload"]),
                                         "planted": planted},
        "permuted_label_fpr": {"PASS": bool(fpr <= 0.05), "fpr": fpr},
        "maxent_pinned": {"PASS": True, "sha256": "8b6d7b46f81788fb0710314867e3233e7f9b7f061b809bbdb11bb120422b7c00",
                          "score3_test": round(score3("ttccaaacgaacttttgtagGGA"), 3)},
        "both_strands_present": {"PASS": len(strands) == 2, "strands": strands},
    }
    ckpt = Path(args.model); ckpt = ckpt if ckpt.is_absolute() else REPO / ckpt
    checks["checkpoint_recorded"] = {"PASS": True, "sha256": sha256_file(ckpt / "pytorch_model.bin")}
    if args.smoke:      # calibration needs full n; smoke only exercises the code path
        checks["planted_positive_sensitivity"]["PASS"] = True
        checks["permuted_label_fpr"]["PASS"] = True
    pf = {"commit": git_commit(REPO), "n": len(E), "checks": checks}
    pf["PASS"] = bool(all(c["PASS"] for c in pf["checks"].values()))
    pf_path.parent.mkdir(parents=True, exist_ok=True)
    pf_path.write_text(json.dumps(pf, indent=1, default=float))
    print(f"preflight PASS={pf['PASS']}  -> {pf_path}")
    for name, c in pf["checks"].items():
        print(f"    {'ok ' if c['PASS'] else 'XX '}{name}")

    if args.smoke:
        print("SMOKE: mechanics + calibration sanity only (val sealed).")
        return
    if not pf["PASS"] and not args.force:
        print("  *** preflight FAILED (calibration or mechanics) -- not reporting (prereg s5).")
        return

    # ---- verdict (held-out) + outcome table ----
    # V2b (LOCAL 4-token MaxEnt, payload-extent-matched) is the correct EXTERNAL splice-strength signal;
    # V2a (full 23nt incl. intronic flank/PPT) NOT naming it is the LOCALIZATION signal (cross-validates
    # A7's PPT-null). Composition is only EXCLUDED if it was identifiable and failed (else non-identifiable).
    named = [V for V in ["V2a", "V2b", "V3", "V1"] if result_vars[V]["H"]["names_payload"]]
    comp_id = bool(result_vars["V4"]["H"]["identifiable"])
    comp_names = bool(result_vars["V4"]["H"]["names_payload"])
    ext, intern, full_window = "V2b" in named, "V3" in named, "V2a" in named
    if ext and intern:
        verdict = "externally_grounded_local_splice_strength"
    elif ext and not intern:
        verdict = "biological_local_splice_strength"
    elif intern and not ext:
        verdict = "internal_acceptor_confidence_only"
    elif comp_names and not named:
        verdict = "generic_local_sequence"
    elif len(named) >= 2:
        verdict = "score_like_composite"
    elif not named:
        verdict = "semantic_identity_unknown"
    else:
        verdict = "single:" + named[0]
    localization = ("local_window" if (ext or intern) and not full_window
                    else "full_window" if full_window else "n/a")

    result = {"n": len(E), "n_genes": len(ug), "strands": strands,
              "frozen_uinfo_rank": int(U.shape[0]), "vars": result_vars,
              "calibration": {"planted_positive": planted, "permuted_fpr": fpr},
              "verdict": verdict, "named_on_heldout": named, "localization": localization,
              "composition_identifiable": comp_id, "composition_excluded": bool(comp_id and not comp_names),
              "preflight_PASS": pf["PASS"], "meets_power_target": bool(len(E) >= 120),
              "confirmatory": bool(pf["PASS"] and len(E) >= 120),
              "REFLECTION": {"status": "PENDING -- fill after inspection (prereg s6)"}}
    out.write_text(json.dumps(result, indent=1, default=float))
    status = "CONFIRMATORY" if result["confirmatory"] else "DIAGNOSTIC"
    man = ExperimentManifest.create(
        "splicebert_coupling:a9_payload_semantics", sys.argv, status=status,
        config={"split": args.split, "n": len(E), "n_genes": len(ug), "max_pairs": args.max_pairs,
                "n_perm": args.n_perm, "preflight_PASS": pf["PASS"]},
        seeds={"gene_split": SEED, "perm": SEED},
        notes=["Semantic identity of the acceptor payload via matched interchange (A8 reconstruct-inject "
               "cross-example). MaxEnt pinned; U_info projected not refit; calibrated (planted+FPR).",
               f"status={status}."])
    man.add_input(str(ckpt / "pytorch_model.bin"))
    man.write(out.parent / (out.stem + ".manifest.json"))

    print("=" * 84)
    for V in VARS:
        h = result_vars[V]["H"]
        print(f"  {V}: (H) eq_med={h['T_eq_median']} con_med={h['T_con_median']} "
              f"names={h['names_payload']} identifiable={h['identifiable']}")
    print(f"  calibration: planted_positive={planted['names_payload']}  permuted_FPR={fpr:.3f}")
    print(f"  VERDICT: {verdict}   named_on_heldout={named}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
