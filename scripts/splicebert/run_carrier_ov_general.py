"""C12 general-OV read -- does the ≈10.63 carrier's reader set WRITE carrier-specific content onto the
GENERAL MLM output (its native target), as opposed to the DONOR readout (measured null by C37/carrier_ov_av)?

Owner decision (2026-07-23): run the CHEAP simple general-OV estimand, then CLOSE C12. This is the
donor-scoped-terminal complement -- a BOUNDED, DIAGNOSTIC read (not a preregistered equivalence test; origin
is settled by C39/C40, which lowers the value of a full powered study). It reuses the exact BAND_P10
deletion + A/V-projection patching of run_carrier_ov_av.py (C37), changing ONLY the readout: instead of the
donor coupling margin, we measure the value main effect on the FULL vocabulary logits at NON-SITE positions.

Estimand (simple single-row value main, the M-1a anti-R23 form, generalised to the general output):
  For each R_1024 reader head h, on the clean input, pin attention at the carrier-DELETED pattern (A=D) and
  compare value INTACT vs value DELETED (V=I vs V=D):
     G_h = E_examples[ mean_{p in NONSITE} ||logits(A=D,V=I)[p] - logits(A=D,V=D)[p]||_2
                                            / ||logits(A=D,V=D)[p]||_2 ]        (relative logit change)
  G_h = 0 iff the head's values are carrier-independent for the general output (a pure router).
Router-negative reference (same pipeline): period-3 L3H1 (BAND=339..345) gives the pipeline's null scale.
Bounded read: the carrier's general OV write is "at the router-null scale" if G_carrier ~ G_router.

DIAGNOSTIC, donor-scoped-complement; NOT a head-local write claim (R24: DEEP L2-L4 readers = within-
intervention conditional mediation). Delta not interpreted as binding (R23). No general-NLL (MPS-wall).
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
SESOI_G = 0.01                     # relative-logit-change SESOI for the bounded read (1% of the logit norm)
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


def general_value_main(ad, task, emb, P0, P_del, layer, head, exs, dev):
    """Per-example relative general-output change from the value main effect (A pinned at D; V=I vs V=D),
    averaged over NON-SITE positions. Returns (per_example_G[list], genes[list])."""
    G, genes = [], []
    with torch.no_grad():
        for i in range(0, len(exs), task.batch):
            ch = exs[i:i + task.batch]
            genes += [x.gene_id for x in ch]
            cl, co, dn = task.pair(ch)
            B, T = cl["input_ids"].shape
            allpos = torch.arange(T, device=dev)[None].expand(B, -1)
            emb.weight.data = P0.clone()
            capI = capture_proj(ad, layer, WH, lambda: ad.model(**cl))     # intact values
            emb.weight.data = P_del.clone()
            capD = capture_proj(ad, layer, WH, lambda: ad.model(**cl))     # carrier-deleted values
            # A pinned at D (query,key from capD); V=I vs V=D
            with contextlib.ExitStack() as st:
                st.enter_context(patch_proj(ad, layer, head, "query", capD["query"], allpos))
                st.enter_context(patch_proj(ad, layer, head, "key", capD["key"], allpos))
                st.enter_context(patch_proj(ad, layer, head, "value", capI["value"], allpos))
                lg_VI = ad.model(**cl).logits.float()
            with contextlib.ExitStack() as st:
                st.enter_context(patch_proj(ad, layer, head, "query", capD["query"], allpos))
                st.enter_context(patch_proj(ad, layer, head, "key", capD["key"], allpos))
                st.enter_context(patch_proj(ad, layer, head, "value", capD["value"], allpos))
                lg_VD = ad.model(**cl).logits.float()
            delta = (lg_VI - lg_VD)                                        # [B,T,V]
            dn_pos = [d + task.off for d in dn]
            for b, x in enumerate(ch):
                acc = (x.next_acceptor_pos + task.off) if x.next_acceptor_pos is not None else -99
                mask = torch.ones(T, dtype=torch.bool, device=dev)
                for pp in (dn_pos[b], dn_pos[b] - 1, dn_pos[b] + 1, acc, acc - 1, acc - 2):
                    if 0 <= pp < T:
                        mask[pp] = False
                dnum = delta[b][mask].norm(dim=-1)                         # per-position L2 change
                dden = lg_VD[b][mask].norm(dim=-1).clamp(min=1e-6)
                G.append(float((dnum / dden).mean()))
    emb.weight.data = P0.clone()
    return G, genes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-heads", type=int, default=8)
    ap.add_argument("--out", default="results/phase-lattice/carrier_ov_general.json")
    args = ap.parse_args()
    t0 = time.time()

    pf = json.loads(PREFLIGHT.read_text())
    R = pf.get("R_1024", [])
    if not R:
        Path(args.out).write_text(json.dumps({"endpoint": "B_R1024_empty", "R_1024": []}, indent=1))
        print("R_1024 empty -> nothing to run.")
        return
    mt = {h["head"]: h["m_T"] for h in pf.get("heads", [])}
    R = sorted(R, key=lambda h: -mt.get(h, 0.0))
    dropped = R[args.max_heads:]; R = R[:args.max_heads]
    heads = [(s, *parse_head(s)) for s in R]

    dev = pick_device(os.environ.get("P3_DEVICE"))
    ad = load_adapter(DEFAULT_MODEL, dev, eager=True)
    task = CouplingTask(ad, batch=8)
    emb = ad.model.bert.embeddings.position_embeddings
    P0 = emb.weight.detach().clone()
    P_del = make_band(P0, BAND_P10)

    n = 24 if args.smoke else args.n
    exs, _ = load_examples(DEFAULT_GTF, DEFAULT_FASTA, "val", n, task=task, tag="c12-ovgen", min_effect=0.3)
    genes_all = sorted({x.gene_id for x in exs})
    print(f"n={len(exs)} genes={len(genes_all)} load {time.time()-t0:.0f}s")

    result = {"carrier_bins": BAND_P10, "SESOI_G": SESOI_G, "R_1024_used": [h[0] for h in heads],
              "R_1024_dropped_for_budget": dropped, "n": len(exs), "smoke": args.smoke,
              "checkpoint": "SpliceBERT.1024nt sha 2ad91428..0149d",
              "commit": subprocess.getoutput("git rev-parse --short HEAD"),
              "readout": "relative general-output logit change ||dlogits||/||logits|| at NON-SITE positions",
              "scope_note": "DIAGNOSTIC bounded read (origin settled C39/C40); general-output complement to the "
                            "donor-scoped C37 null; within-intervention (R24); NOT a head-local write claim.",
              "heads": {}}

    # router-negative reference (period-3 L3H1)
    rg, rgenes = general_value_main(ad, task, emb, P0, make_band(P0, P3_BAND), 3, 1, exs, dev)
    rlo, rhi = gboot(rg, rgenes)
    result["router_negative_L3H1_period3"] = {"G_mean": float(np.mean(rg)), "ci": [rlo, rhi]}
    print(f"router-negative L3H1: G {np.mean(rg):.4f} [{rlo:.4f},{rhi:.4f}]")

    fired = []
    for s, l, h in heads:
        g, gg = general_value_main(ad, task, emb, P0, P_del, l, h, exs, dev)
        lo, hi = gboot(g, gg)
        # "writes generally" iff G exceeds the router-null scale by > SESOI_G (CI-low above router mean + SESOI)
        writes = bool(lo > np.mean(rg) + SESOI_G)
        result["heads"][s] = {"G_mean": float(np.mean(g)), "ci": [lo, hi], "writes_general": writes}
        if writes:
            fired.append(s)
        print(f"  {s}: G {np.mean(g):.4f} [{lo:.4f},{hi:.4f}]{'  <== WRITES GENERAL' if writes else ''}")

    endpoint = ("A_carrier_writes_general_output" if fired
                else "A_null_carrier_readers_route_not_write_general_output")
    result["endpoint"] = endpoint
    result["fired_heads"] = fired

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    ExperimentManifest.create(
        "c12:carrier_ov_general", sys.argv, repo=REPO, status=endpoint,
        config={"carrier_bins": BAND_P10, "SESOI_G": SESOI_G, "heads": [h[0] for h in heads], "n": len(exs)},
        seeds={"PYTHONHASHSEED": 0},
        notes=["C12 general-OV bounded read: value main (simple DI-DD) on the GENERAL MLM output at the "
               "≈10.63 readers vs the period-3 router-null scale. Owner: run cheap then close C12. Diagnostic; "
               "no general-NLL (MPS-wall); within-intervention (R24)."],
    ).write(Path(args.out).with_suffix(".manifest.json"))
    print(f"\nENDPOINT: {endpoint}   fired={fired or 'none'}   TOTAL {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
