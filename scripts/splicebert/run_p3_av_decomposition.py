"""P3.3 -- A-vs-V routing decomposition of the period-3 carrier's coupling contribution at L3H1.

Prereg: PREREGISTRATION_P3_CARRIER_CAUSAL_ROLE.md (v2). Does the carrier's contribution to donor
coupling route through L3H1's ATTENTION PATTERN (A) or its VALUES (V)?  On the DELETED base network,
patch L3H1 head-1's query,key to carrier-x captures (=> A_x) and value to carrier-y captures (=> V_y).

  routing main = C(A_I,V_D) - C(A_D,V_D)      value main = C(A_D,V_I) - C(A_D,V_D)
  Delta_bind   = C(A_I,V_I) - C(A_I,V_D) - C(A_D,V_I) + C(A_D,V_D)   [REPORTED, NOT "binding" -- R23:
                 architecture-forced by the bilinear O=A*V; needs an importance-matched null to interpret]

Axis is "A vs V" (NOT address vs payload -- V carries position, R15). Conditional mediation only.
Plumbing anchors validated in smoke (p3_3_av_smoke): M4 diagonal, off-diagonal native, head-identity, all 0.0.
Per-example PAIRED contrasts, gene-clustered bootstrap; discovery/confirm gene split; health per cell.
"""
from __future__ import annotations
import argparse, contextlib, json, os, subprocess, sys, time
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "src")
from gfmcd.splicing.circuit import (
    CouplingTask, load_adapter, load_examples, pick_device, gboot, capture_proj, patch_proj,
    DEFAULT_GTF, DEFAULT_FASTA, DEFAULT_MODEL,
)
from gfmcd.splicing.coupling import stable_rng

BAND = list(range(339, 346))
L, Hd, WH = 3, 1, ["query", "key", "value"]


def make_band(P0, bins):
    P = P0.detach().float().cpu().numpy()
    mu = P.mean(0, keepdims=True)
    F = np.fft.rfft(P - mu, axis=0); F[bins] = 0.0
    return torch.tensor(np.fft.irfft(F, n=P.shape[0], axis=0) + mu, dtype=P0.dtype, device=P0.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results/p3_av_decomposition.json")
    args = ap.parse_args()
    t0 = time.time()
    dev = pick_device(os.environ.get("P3_DEVICE"))
    ad = load_adapter(DEFAULT_MODEL, dev, eager=True)
    task = CouplingTask(ad, batch=8)
    emb = ad.model.bert.embeddings.position_embeddings
    P0 = emb.weight.detach().clone()
    P_del = make_band(P0, BAND)

    n = 24 if args.smoke else args.n
    exs, _ = load_examples(DEFAULT_GTF, DEFAULT_FASTA, "val", n, task=task, tag="p3-av", min_effect=0.3)
    genes_all = sorted({x.gene_id for x in exs})
    disc = set(genes_all[::2]); conf = set(genes_all[1::2])
    print(f"n={len(exs)} genes={len(genes_all)} load {time.time()-t0:.0f}s")

    def margin_forward(enc, dn, patches):
        B, T = enc["input_ids"].shape
        allpos = torch.arange(T, device=dev)[None].expand(B, -1)
        with contextlib.ExitStack() as st:
            for which, src in patches:
                st.enter_context(patch_proj(ad, L, Hd, which, src, allpos))
            return task.margin(ad.model(**enc).logits.float(), dn).detach()

    def health_cell(ch, patches_of):
        """global masked-probe accuracy under a cell's patch (clean encoding)."""
        r = stable_rng("h-av", 0)
        dn = [x.donor_pos for x in ch]
        enc = task.encode([x.sequence for x in ch], dn)
        probes = []
        for j, x in enumerate(ch):
            ps = [p for p in sorted(r.sample(range(10, len(x.sequence) - 10), 30))
                  if abs(p - x.donor_pos) > 2 and abs(p - (x.next_acceptor_pos or -99)) > 3]
            for p in ps:
                enc["input_ids"][j, p + task.off] = task.mask_id
            probes.append(ps)
        B, T = enc["input_ids"].shape
        allpos = torch.arange(T, device=dev)[None].expand(B, -1)
        with contextlib.ExitStack() as st:
            for which, src in patches_of(enc):
                st.enter_context(patch_proj(ad, L, Hd, which, src, allpos))
            lg = ad.model(**enc).logits.float()
        acc = []
        for j, x in enumerate(ch):
            if not probes[j]:
                continue
            pred = lg[j, [p + task.off for p in probes[j]]].argmax(-1)
            tgt = torch.tensor([task.bid[x.sequence[p]] for p in probes[j]], device=pred.device)
            acc.append((pred == tgt).float().mean().item())
        return float(np.mean(acc))

    cells = {k: [] for k in ["DD", "ID", "DI", "II"]}
    genes = []
    health = {k: [] for k in ["DD", "ID", "DI", "II"]}
    with torch.no_grad():
        for i in range(0, len(exs), task.batch):
            ch = exs[i:i + task.batch]
            genes += [x.gene_id for x in ch]
            cl, co, dn = task.pair(ch)
            emb.weight.data = P0.clone()
            capIc = capture_proj(ad, L, WH, lambda: ad.model(**cl))
            capIk = capture_proj(ad, L, WH, lambda: ad.model(**co))
            emb.weight.data = P_del.clone()
            capDc = capture_proj(ad, L, WH, lambda: ad.model(**cl))
            capDk = capture_proj(ad, L, WH, lambda: ad.model(**co))
            src = {"I": (capIc, capIk), "D": (capDc, capDk)}

            def cell(xA, yV):
                cA, kA = src[xA]; cV, kV = src[yV]
                mc = margin_forward(cl, dn, [("query", cA["query"]), ("key", cA["key"]), ("value", cV["value"])])
                mk = margin_forward(co, dn, [("query", kA["query"]), ("key", kA["key"]), ("value", kV["value"])])
                return (mc - mk)

            for k, (xA, yV) in {"DD": ("D", "D"), "ID": ("I", "D"), "DI": ("D", "I"), "II": ("I", "I")}.items():
                cells[k] += cell(xA, yV).tolist()
            if not args.smoke:
                for k, (xA, yV) in {"DD": ("D", "D"), "ID": ("I", "D"), "DI": ("D", "I"), "II": ("I", "I")}.items():
                    cA, kA = src[xA]; cV, kV = src[yV]
                    health[k].append(health_cell(ch, lambda enc, cA=cA, cV=cV:
                                                 [("query", cA["query"]), ("key", cA["key"]), ("value", cV["value"])]))
    emb.weight.data = P0.clone()
    for k in cells:
        cells[k] = np.asarray(cells[k])
    genes = np.asarray(genes)

    def summ(name, S):
        m = np.array([g in S for g in genes])
        gg = genes[m]
        DD, ID, DI, II = (cells[k][m] for k in ["DD", "ID", "DI", "II"])
        routing = ID - DD; value = DI - DD; dbind = II - ID - DI + DD
        out = {"n": int(m.sum())}
        for nm, v in [("cell_DD", DD), ("cell_ID", ID), ("cell_DI", DI), ("cell_II", II),
                      ("routing_main", routing), ("value_main", value), ("delta_bind", dbind)]:
            lo, hi = gboot(v.tolist(), gg.tolist())
            out[nm] = {"mean": float(np.mean(v)), "ci": [lo, hi]}
        return out

    result = {"config": {"n": len(exs), "smoke": args.smoke, "band": [BAND[0], BAND[-1]]},
              "provenance": {"commit": subprocess.getoutput("git rev-parse --short HEAD"),
                             "pythonhashseed": os.environ.get("PYTHONHASHSEED"), "device": dev,
                             "checkpoint": "SpliceBERT.1024nt sha 2ad91428..0149d"},
              "note": "R23: axis is A-vs-V (not address/payload, R15); delta_bind reported NOT interpreted "
                      "as binding (architecture-forced, needs importance-matched null). Conditional mediation."}
    if health["DD"]:
        result["health_global"] = {k: float(np.mean(health[k])) for k in health}
    result["discovery"] = summ("discovery", disc)
    if not args.smoke:
        result["confirm"] = summ("confirm", conf)

    for split in (["discovery", "confirm"] if not args.smoke else ["discovery"]):
        r = result[split]
        print(f"\n=== {split.upper()} (n={r['n']}) ===")
        for k in ["cell_DD", "cell_ID", "cell_DI", "cell_II"]:
            print(f"  {k}: {r[k]['mean']:+.3f} [{r[k]['ci'][0]:+.3f},{r[k]['ci'][1]:+.3f}]")
        print(f"  ROUTING main (A_I,V_D-A_D,V_D): {r['routing_main']['mean']:+.3f} "
              f"[{r['routing_main']['ci'][0]:+.3f},{r['routing_main']['ci'][1]:+.3f}]")
        print(f"  VALUE   main (A_D,V_I-A_D,V_D): {r['value_main']['mean']:+.3f} "
              f"[{r['value_main']['ci'][0]:+.3f},{r['value_main']['ci'][1]:+.3f}]")
        print(f"  delta_bind (reported, R23):     {r['delta_bind']['mean']:+.3f} "
              f"[{r['delta_bind']['ci'][0]:+.3f},{r['delta_bind']['ci'][1]:+.3f}]")
    if result.get("health_global"):
        print("  health(global): " + " ".join(f"{k}={v:.2f}" for k, v in result["health_global"].items()))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}  TOTAL {time.time()-t0:.0f}s\n{'SMOKE_OK' if args.smoke else 'RUN_OK'}")


if __name__ == "__main__":
    main()
