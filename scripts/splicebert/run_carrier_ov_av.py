"""B1 OV arm — does the ≈10.63 carrier change what its post-softmax reader heads WRITE (value main
effect), and does that write reach the donor readout / coupling?  (PREREGISTRATION_B1_..._V2.md §1 B1-OV.)

Generalizes the C35 (P3.3) A-vs-V decomposition (run_p3_av_decomposition.py) from the single period-3 head
L3H1 to the R_1024 reader set (read from carrier_preflight_r.json), at BAND_P10 = [94..100].  The estimand is
the SIMPLE single-row value main (the M-1a fix):

    Mv_h (value main)  = C(A=D10, V=I) - C(A=D10, V=D10) = A_D * (V_I - V_D)   [= 0 iff values carrier-indep]
    routing main       = C(A=I,   V=D10) - C(A=D10, V=D10)
    Delta_bind         = C(II) - C(ID) - C(DI) + C(DD)     [architecture-forced, R23 -- reported, not interpreted]

    ⚠️ the SYMMETRIC two-row main 1/2[(DI-DD)+(II-ID)] = Mv_h + 1/2*Delta_bind re-imports half the forced term
       (the M-1a catch); M-simplemain verifies the code uses the SIMPLE form (on the router it gives -0.005,
       not +0.073).

Router-negative (M-vmain, must-fix #2): the SAME pipeline on period-3 L3H1 (BAND=339..345) must return a value
main within +-SESOI_v (the C35 anchor -0.005) -- proving the pipeline CAN null a known router at the readout
scale.  SESOI_v pre-declared >= 0.02 (the confirm-split router-null scale; must-fix #1b/#7).

Scope (R24 / must-fix #7):  R_1024 is entirely L2/L3/L4 (no L0 readers), so the value main at these DEEP heads
is a WITHIN-INTERVENTION conditional mediation (exactly as C35 measured at L3H1) -- it does NOT exclude
upstream carrier effects on payload construction and is NOT a head-local "write" claim.  Donor-scoped.

The covariance-matched null (must-fix #2) and PP-ovplant are the CALIBRATION for a POSITIVE; they are the
expensive net-new pieces and are deferred to a TARGETED follow-up (--null) run ONLY for heads whose core value
main fires (CI-low > SESOI_v).  The modal expected outcome (given the behavioural null + C35) is A-null: every
head's value main within +-SESOI_v -> the carrier's readers route but do not write onto coupling.

⚠️ UNIT-UNTESTED until --smoke passes on a checkpoint-bearing checkout.
"""
from __future__ import annotations
import argparse, contextlib, json, os, subprocess, sys, time
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    CouplingTask, load_adapter, load_examples, pick_device, gboot, capture_proj, patch_proj,
    DEFAULT_GTF, DEFAULT_FASTA, DEFAULT_MODEL,
)

BAND_P10 = list(range(94, 101))
P3_BAND = list(range(339, 346))
WH = ["query", "key", "value"]
SESOI_v = 0.02
PREFLIGHT = REPO / "results/phase-lattice/carrier_preflight_r.json"


def make_band(P0, bins):
    P = P0.detach().float().cpu().numpy()
    mu = P.mean(0, keepdims=True)
    F = np.fft.rfft(P - mu, axis=0)
    F[bins] = 0.0
    return torch.tensor(np.fft.irfft(F, n=P.shape[0], axis=0) + mu, dtype=P0.dtype, device=P0.device)


def parse_head(s):
    l, h = s[1:].split("H")
    return int(l), int(h)


def av_cells(ad, task, emb, P0, P_del, layer, head, exs, dev):
    """C35's per-head A-vs-V 2x2 at the coupling readout, generalised to (layer, head).
    Returns dict of per-example cell arrays {DD,ID,DI,II} and the gene list."""
    cells = {k: [] for k in ("DD", "ID", "DI", "II")}
    genes = []
    with torch.no_grad():
        for i in range(0, len(exs), task.batch):
            ch = exs[i:i + task.batch]
            genes += [x.gene_id for x in ch]
            cl, co, dn = task.pair(ch)
            B, T = cl["input_ids"].shape
            allpos = torch.arange(T, device=dev)[None].expand(B, -1)
            emb.weight.data = P0.clone()
            capIc = capture_proj(ad, layer, WH, lambda: ad.model(**cl))
            capIk = capture_proj(ad, layer, WH, lambda: ad.model(**co))
            emb.weight.data = P_del.clone()
            capDc = capture_proj(ad, layer, WH, lambda: ad.model(**cl))
            capDk = capture_proj(ad, layer, WH, lambda: ad.model(**co))
            src = {"I": (capIc, capIk), "D": (capDc, capDk)}

            def cell(xA, yV):
                cA, kA = src[xA]; cV, kV = src[yV]
                with contextlib.ExitStack() as st:
                    st.enter_context(patch_proj(ad, layer, head, "query", cA["query"], allpos))
                    st.enter_context(patch_proj(ad, layer, head, "key", cA["key"], allpos))
                    st.enter_context(patch_proj(ad, layer, head, "value", cV["value"], allpos))
                    mc = task.margin(ad.model(**cl).logits.float(), dn).detach()
                with contextlib.ExitStack() as st:
                    st.enter_context(patch_proj(ad, layer, head, "query", kA["query"], allpos))
                    st.enter_context(patch_proj(ad, layer, head, "key", kA["key"], allpos))
                    st.enter_context(patch_proj(ad, layer, head, "value", kV["value"], allpos))
                    mk = task.margin(ad.model(**co).logits.float(), dn).detach()
                return (mc - mk)

            for k, (xA, yV) in {"DD": ("D", "D"), "ID": ("I", "D"), "DI": ("D", "I"), "II": ("I", "I")}.items():
                cells[k] += cell(xA, yV).tolist()
    emb.weight.data = P0.clone()
    return cells, genes


def summarise(cells, genes):
    DD, ID, DI, II = (np.asarray(cells[k]) for k in ("DD", "ID", "DI", "II"))
    value = DI - DD              # SIMPLE single-row value main (M-1a)
    routing = ID - DD
    dbind = II - ID - DI + DD
    symmetric = 0.5 * ((DI - DD) + (II - ID))   # reported for the M-simplemain identity check
    out = {"n": len(genes)}
    for nm, v in [("value_main", value), ("routing_main", routing), ("delta_bind", dbind),
                  ("symmetric_main", symmetric)]:
        lo, hi = gboot(v.tolist(), genes)
        out[nm] = {"mean": float(np.mean(v)), "ci": [lo, hi]}
    # M-simplemain arithmetic: symmetric == value + 0.5*delta_bind (identity; must hold to <1e-9)
    out["M_simplemain_gap"] = float(np.max(np.abs(symmetric - (value + 0.5 * dbind))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-heads", type=int, default=8, help="cap R_1024 by preflight m_T for the MPS budget")
    ap.add_argument("--out", default="results/phase-lattice/carrier_ov_av.json")
    args = ap.parse_args()
    t0 = time.time()

    pf = json.loads(PREFLIGHT.read_text())
    R = pf.get("R_1024", [])
    if not R:
        print("R_1024 is empty in carrier_preflight_r.json -> OV arm is Endpoint B (nothing to run).")
        Path(args.out).write_text(json.dumps({"endpoint": "B_R1024_empty", "R_1024": []}, indent=1))
        return
    # cap by preflight m_T for the budget (report the drop)
    mt = {h["head"]: h["m_T"] for h in pf.get("heads", [])}
    R = sorted(R, key=lambda h: -mt.get(h, 0.0))
    dropped = R[args.max_heads:]
    R = R[:args.max_heads]
    heads = [(s, *parse_head(s)) for s in R]
    print(f"R_1024 (top {len(heads)} by m_T): {[h[0] for h in heads]}   dropped: {dropped}")

    dev = pick_device(os.environ.get("P3_DEVICE"))
    ad = load_adapter(DEFAULT_MODEL, dev, eager=True)
    task = CouplingTask(ad, batch=8)
    emb = ad.model.bert.embeddings.position_embeddings
    P0 = emb.weight.detach().clone()
    P_del = make_band(P0, BAND_P10)

    n = 24 if args.smoke else args.n
    exs, _ = load_examples(DEFAULT_GTF, DEFAULT_FASTA, "val", n, task=task, tag="b1-ov", min_effect=0.3)
    genes_all = sorted({x.gene_id for x in exs})
    print(f"n={len(exs)} genes={len(genes_all)} load {time.time()-t0:.0f}s")

    result = {"carrier_bins": BAND_P10, "SESOI_v": SESOI_v, "R_1024_used": [h[0] for h in heads],
              "R_1024_dropped_for_budget": dropped, "n": len(exs), "smoke": args.smoke,
              "checkpoint": "SpliceBERT.1024nt sha 2ad91428..0149d",
              "commit": subprocess.getoutput("git rev-parse --short HEAD"),
              "scope_note": "value main at DEEP (L2-L4) readers = within-intervention conditional mediation "
                            "(C35/R24); NOT a head-local write claim; donor-scoped; Delta_bind reported not "
                            "interpreted (R23).", "heads": {}}

    # ── ROUTER-NEGATIVE (M-vmain): period-3 L3H1 through the SAME pipeline must null ──────────────
    print("router-negative: L3H1 / period-3 ...")
    rc, rg = av_cells(ad, task, emb, P0, make_band(P0, P3_BAND), 3, 1, exs, dev)
    router = summarise(rc, rg)
    result["router_negative_L3H1_period3"] = router
    m_vmain_pass = bool(abs(router["value_main"]["mean"]) <= SESOI_v
                        and router["M_simplemain_gap"] < 1e-9)
    print(f"  router value_main {router['value_main']['mean']:+.4f} "
          f"[{router['value_main']['ci'][0]:+.4f},{router['value_main']['ci'][1]:+.4f}]  "
          f"delta_bind {router['delta_bind']['mean']:+.4f}  M-simplemain_gap {router['M_simplemain_gap']:.1e}")

    # ── R_1024 heads: value main at BAND_P10 ─────────────────────────────────────────────────────
    fired = []
    for s, l, h in heads:
        c, g = av_cells(ad, task, emb, P0, P_del, l, h, exs, dev)
        summ = summarise(c, g)
        result["heads"][s] = summ
        lo = summ["value_main"]["ci"][0]
        writes = bool(lo > SESOI_v)
        summ["writes_onto_coupling"] = writes
        if writes:
            fired.append(s)
        print(f"  {s}: value_main {summ['value_main']['mean']:+.4f} "
              f"[{summ['value_main']['ci'][0]:+.4f},{summ['value_main']['ci'][1]:+.4f}]  "
              f"routing {summ['routing_main']['mean']:+.4f}  dbind {summ['delta_bind']['mean']:+.4f}"
              f"{'  <== WRITES' if writes else ''}")

    # ── endpoint (donor-scoped; coupling positivity is the behavioural arm) ───────────────────────
    if not m_vmain_pass:
        endpoint = "B_router_negative_failed"
    elif fired:
        endpoint = "A_OV_candidate_heads_write_run_covariance_null_next"
    else:
        endpoint = "A_null_carrier_readers_route_not_write_onto_coupling"
    result["endpoint"] = endpoint
    result["fired_heads"] = fired
    result["M_vmain_pass"] = m_vmain_pass

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    ExperimentManifest.create(
        "b1:carrier_ov_av", sys.argv, repo=REPO, status=endpoint,
        config={"carrier_bins": BAND_P10, "SESOI_v": SESOI_v, "heads": [h[0] for h in heads], "n": len(exs)},
        seeds={"PYTHONHASHSEED": 0},
        notes=["B1 OV arm: value main (simple DI-DD) at the ≈10.63 post-softmax readers, generalising C35. "
               "Router-negative L3H1/period-3 gates M-vmain. Covariance-matched null deferred to --null "
               "follow-up for fired heads only. Deep-reader value main = within-intervention mediation (R24)."],
    ).write(Path(args.out).with_suffix(".manifest.json"))
    print(f"\nENDPOINT: {endpoint}   fired={fired or 'none'}")
    print(f"wrote {args.out}  TOTAL {time.time()-t0:.0f}s  {'SMOKE_OK' if args.smoke else 'RUN_OK'}")


if __name__ == "__main__":
    main()
