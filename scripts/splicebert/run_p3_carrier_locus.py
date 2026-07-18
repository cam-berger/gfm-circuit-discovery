"""P3.2 -- the period-3 carrier's causal LOCUS in donor coupling (confirmatory).

Prereg: PREREGISTRATION_P3_CARRIER_CAUSAL_ROLE.md (v2). Question (Cam): WHERE must the carrier be
present?  Restore the carrier delta (P_intact - P_deleted) in the DELETED model at position-selective
windows and measure normalized recovery of coupling R_C(a) = (C_a - C_del)/(C_intact - C_del).

Arms (window half-width w): donor, acceptor, both, phase-matched-irrelevant (specificity null),
decoy-acceptor (splice-like content, wrong identity), all (forced memorial), none (=deleted).

Memorials (failable, on-path): M1 C(D)/C(I)<=0.05; construction P_del+delta==P_intact; none==deleted;
all==intact (forced, M5); MONOTONE dose-response R_C(w) (the real selective-restoration test).
Health-gated per hybrid (R13).  Gene-clustered ratio CI (gboot_ratio, never mean-of-ratios).
Discovery/confirm gene split (untouched-gene confirmation, R22).

Reused plumbing validated in smoke (p3_2_restoration_pilot).  Compute: MPS.  Deterministic (PYTHONHASHSEED).
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
os.environ.setdefault("PYTHONHASHSEED", "0")
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "src")
from gfmcd.splicing.circuit import (
    CouplingTask, load_adapter, load_examples, pick_device, gboot_ratio,
    DEFAULT_GTF, DEFAULT_FASTA, DEFAULT_MODEL,
)
from gfmcd.splicing.address import phase_matched_decoy
from gfmcd.splicing.coupling import stable_rng

BAND = list(range(339, 346))
REPO = Path(__file__).resolve().parents[2]


def make_band(P0, bins):
    P = P0.detach().float().cpu().numpy()
    mu = P.mean(0, keepdims=True)
    F = np.fft.rfft(P - mu, axis=0); F[bins] = 0.0
    return torch.tensor(np.fft.irfft(F, n=P.shape[0], axis=0) + mu, dtype=P0.dtype, device=P0.device)


@contextmanager
def restore_windows(ad, delta, win_tokens):
    emb = ad.model.bert.embeddings.position_embeddings

    def hook(_m, _i, out):
        B = len(win_tokens)
        o = out.expand(B, -1, -1).clone() if out.shape[0] == 1 else out.clone()
        for b, toks in enumerate(win_tokens):
            if toks.numel():
                o[b, toks] = o[b, toks] + delta[toks].to(o.dtype)
        return o

    h = emb.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


def build_win(x, off, T, kind, w):
    """Token indices to restore for one example; None if the arm is unavailable for this example."""
    N = len(x.sequence)
    d, a = x.donor_pos + off, x.next_acceptor_pos + off

    def clip(lo, hi):
        return [t for t in range(lo, hi + 1) if off <= t < off + N]

    if kind == "none":
        return []
    if kind == "all":
        return list(range(off, off + N))
    if kind == "donor":
        return clip(d - w, d + w)
    if kind == "acceptor":
        return clip(a - w, a + w)
    if kind == "both":
        return sorted(set(clip(d - w, d + w) + clip(a - w, a + w)))
    if kind == "decoy_acceptor":
        dec = phase_matched_decoy(x)
        return clip(dec + off - w, dec + off + w) if dec is not None else None
    if kind == "phase_irrelevant":
        phase = (a - d) % 3
        agtok = {p + off for p in x.other_ag}
        cands = [c for c in range(off + w, off + N - w)
                 if (c - d) % 3 == phase and (c < d - 3 * w or c > a + 3 * w)
                 and abs(c - d) > 3 * w and abs(c - a) > 3 * w
                 and not any(abs(c - t) <= w for t in agtok)]
        if not cands:
            return None
        c = min(cands, key=lambda c: abs(abs(c - d) - (a - d)))
        return clip(c - w, c + w)
    raise ValueError(kind)


def _wt(ch, off, T, kind, w, dev):
    toks, avail = [], []
    for x in ch:
        t = build_win(x, off, T, kind, w)
        avail.append(t is not None)
        toks.append(torch.tensor(sorted(set(t or [])), dtype=torch.long, device=dev))
    return toks, avail


def coupling_restore(ad, task, exs, delta, kind, w, off, T):
    out = []
    with torch.no_grad():
        for i in range(0, len(exs), task.batch):
            ch = exs[i:i + task.batch]
            wt, av = _wt(ch, off, T, kind, w, delta.device)
            cl, co, dn = task.pair(ch)
            with restore_windows(ad, delta, wt):
                mc = task.margin(ad.model(**cl).logits.float(), dn)
                mk = task.margin(ad.model(**co).logits.float(), dn)
            d = (mc - mk).tolist()
            out += [d[j] if av[j] else float("nan") for j in range(len(ch))]
    return np.asarray(out)


def health_restore(ad, task, exs, delta, kind, w, off, T, n_probe=30, seed=0):
    r = stable_rng(f"h-{kind}-{w}", seed)
    acc = []
    with torch.no_grad():
        for i in range(0, len(exs), task.batch):
            ch = exs[i:i + task.batch]
            dn = [x.donor_pos for x in ch]
            enc = task.encode([x.sequence for x in ch], dn)
            probes = []
            for j, x in enumerate(ch):
                ps = sorted(r.sample(range(10, len(x.sequence) - 10), n_probe))
                ps = [p for p in ps if abs(p - x.donor_pos) > 2 and abs(p - (x.next_acceptor_pos or -99)) > 3]
                for p in ps:
                    enc["input_ids"][j, p + off] = task.mask_id
                probes.append(ps)
            wt, _ = _wt(ch, off, T, kind, w, delta.device)
            with restore_windows(ad, delta, wt):
                lg = ad.model(**enc).logits.float()
            for j, x in enumerate(ch):
                if not probes[j]:
                    continue
                pred = lg[j, [p + off for p in probes[j]]].argmax(-1)
                tgt = torch.tensor([task.bid[x.sequence[p]] for p in probes[j]], device=pred.device)
                acc.append((pred == tgt).float().mean().item())
    return float(np.mean(acc))


def RC(Ca, Cd, Ci):
    m = ~np.isnan(Ca)
    num = (Ca[m] - Cd[m]).tolist()
    den = (Ci[m] - Cd[m]).tolist()
    return float(np.sum(num) / max(np.sum(den), 1e-9)), num, den, m


def split_genes(exs):
    genes = sorted({x.gene_id for x in exs})
    disc = set(genes[::2]); conf = set(genes[1::2])
    return ([x for x in exs if x.gene_id in disc], [x for x in exs if x.gene_id in conf])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--w", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results/p3_carrier_locus.json")
    ap.add_argument("--gtf", default=DEFAULT_GTF); ap.add_argument("--fasta", default=DEFAULT_FASTA)
    args = ap.parse_args()
    t0 = time.time()
    dev = pick_device(os.environ.get("P3_DEVICE"))
    ad = load_adapter(DEFAULT_MODEL, dev, eager=True)
    task = CouplingTask(ad, batch=8)
    emb = ad.model.bert.embeddings.position_embeddings
    P0 = emb.weight.detach().clone()
    P_del = make_band(P0, BAND)
    delta = (P0 - P_del).to(dev)
    off, T = ad.prefix_tokens, P0.shape[0]

    con_err = float((P_del + delta - P0).abs().max())
    n = 24 if args.smoke else args.n
    exs, _ = load_examples(args.gtf, args.fasta, "val", n, task=task, tag="p3-locus", min_effect=0.3)
    disc, conf = split_genes(exs)
    print(f"n={len(exs)} disc={len(disc)} conf={len(conf)} "
          f"genes={len(set(x.gene_id for x in exs))} T={T} load {time.time()-t0:.0f}s")

    arms = ["none", "donor", "acceptor", "both", "phase_irrelevant", "decoy_acceptor", "all"]
    doses = [2, 4, 8, 16, 32]
    result = {"config": {"n": len(exs), "w": args.w, "band": [BAND[0], BAND[-1]], "smoke": args.smoke,
                         "n_disc": len(disc), "n_conf": len(conf)},
              "provenance": {"commit": subprocess.getoutput("git rev-parse --short HEAD"),
                             "pythonhashseed": os.environ.get("PYTHONHASHSEED"), "device": dev,
                             "checkpoint": "SpliceBERT.1024nt sha 2ad91428..0149d"},
              "memorials": {"construction_P_hybrid_max_abs_err": con_err}}

    def run_split(name, S):
        genes = [x.gene_id for x in S]
        Ci = task.coupling(S)
        try:
            emb.weight.data = P_del.clone()
            Cd = task.coupling(S)
            armC = {k: coupling_restore(ad, task, S, delta, k, args.w, off, T) for k in arms}
            dose = {w: coupling_restore(ad, task, S, delta, "both", w, off, T) for w in doses}
            # health for interpreted states: deleted(none), primary hybrid(both@w), intact(all)
            health = {k: health_restore(ad, task, S, delta, k, args.w, off, T)
                      for k in (["none", "both", "all"] if not args.smoke else [])}
        finally:
            emb.weight.data = P0.clone()
        ci, cd = float(np.nanmean(Ci)), float(np.nanmean(Cd))
        out = {"C_intact": ci, "C_deleted": cd, "denom": ci - cd,
               "M1_delete_collapses": bool(cd / max(ci, 1e-9) <= 0.05),
               "none_eq_deleted_maxerr": float(np.nanmax(np.abs(armC["none"] - Cd))),
               "all_eq_intact_maxerr": float(np.nanmax(np.abs(armC["all"] - Ci))),
               "arms": {}, "dose": {}, "health": health}
        for k in arms:
            r, num, den, m = RC(armC[k], Cd, Ci)
            lo, hi = gboot_ratio(num, den, [genes[i] for i in np.where(m)[0]]) if m.sum() else (float("nan"),) * 2
            out["arms"][k] = {"R_C": r, "ci": [lo, hi], "n": int(m.sum()),
                              "C_mean": float(np.nanmean(armC[k]))}
        rc_dose = []
        for w in doses:
            r, *_ = RC(dose[w], Cd, Ci); rc_dose.append(r)
            out["dose"][w] = r
        out["monotone"] = bool(all(rc_dose[i] <= rc_dose[i + 1] + 0.05 for i in range(len(rc_dose) - 1)))
        if health:
            chance, hi_acc = 0.25, health["all"]
            out["health_kept"] = {k: (health[k] - chance) / max(hi_acc - chance, 1e-9) for k in health}
            out["health_gate_pass"] = bool(min(out["health_kept"].values()) >= 0.5)
        return out

    result["discovery"] = run_split("discovery", disc)
    if not args.smoke:
        result["confirm"] = run_split("confirm", conf)

    # report
    print(f"\nconstruction P_hybrid max|Δ|={con_err:.2e} {'PASS' if con_err<1e-5 else 'FAIL'}")
    for split in (["discovery", "confirm"] if not args.smoke else ["discovery"]):
        r = result[split]
        print(f"\n=== {split.upper()}  C_intact={r['C_intact']:+.3f} C_deleted={r['C_deleted']:+.3f} "
              f"denom={r['denom']:+.3f}  M1={r['M1_delete_collapses']} ===")
        print(f"  none==deleted {r['none_eq_deleted_maxerr']:.1e}  all==intact {r['all_eq_intact_maxerr']:.1e}  "
              f"monotone={r['monotone']}")
        if r.get("health"):
            print(f"  health kept: " + " ".join(f"{k}={v:.2f}" for k, v in r["health_kept"].items())
                  + f"  gate={r['health_gate_pass']}")
        for k in arms:
            a = r["arms"][k]
            print(f"  {k:16s} R_C={a['R_C']:+.3f} [{a['ci'][0]:+.3f},{a['ci'][1]:+.3f}] (n={a['n']})")
        print(f"  dose R_C(both,w): " + " ".join(f"w{w}:{r['dose'][w]:+.3f}" for w in doses))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}  TOTAL {time.time()-t0:.0f}s\n{'SMOKE_OK' if args.smoke else 'RUN_OK'}")


if __name__ == "__main__":
    main()
