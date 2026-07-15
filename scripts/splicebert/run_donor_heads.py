#!/usr/bin/env python
"""RESOLVE L4/L5 ATTENTION INTO HEADS -- router vs transporter.  (donor-graph 1c)

C17: L3H1 delivers U_info, assembled at MLP3/MLP4, localized to donor+/-5.  readout_path
credited the layer-4 and layer-5 ATTENTION blocks (whole-block) with carrying the coupling to
the donor readout.  Which heads, and how?

Part 1 (which heads) -- NOISING necessity: replace each L4/L5 head's clean output with the
value it takes when the acceptor is destroyed (its corrupt output), in the CLEAN run.  A head
whose noising drops the coupling carries the acceptor-dependent signal.  Noising-to-corrupt,
NOT mean-ablation, so a head that is merely present on the path (constant relay) does not score
(R12/R13).

Part 2 (how) -- for the top heads, noise QUERY+KEY (routing) vs VALUE (content): a router's
attention pattern carries it (QK), a transporter carries content (OV/value).

Prereg (FROZEN): studies/splicebert_partner_selection/PREREGISTRATION_DONOR_HEADS.md
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from contextlib import ExitStack  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    ADDRESSER, DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, READER, CouplingTask, ablate,
    capture_head_outputs, capture_proj, gboot_ratio, load_adapter, load_examples,
    patch_heads, patch_proj, pick_device, ratio_of_means,
)

L1H12 = (1, 12)
SEED = 0
TOP_K = 3            # heads taken into the router/transporter split


def hkey(h):
    return f"L{h[0]}H{h[1]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="results/splicebert_coupling/donor_heads.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=True)
    task = CouplingTask(ad, batch=args.batch)

    L45 = [(l, h) for l in (4, 5) for h in range(ad.num_heads)]
    REF = [READER, ADDRESSER, L1H12]                 # sanity references (known reader/addresser)
    SCAN = REF + L45

    X, _ = load_examples(args.gtf, args.fasta, args.split, args.n, task=task,
                         tag=f"heads-{args.split}", min_effect=None)
    genes = [x.gene_id for x in X]
    print(f"{len(X)} unconditional {args.split} exons   device={dev}   scanning {len(SCAN)} heads")

    # ---- Part 1: noising necessity scan ------------------------------------------------------
    m_cl, m_co = [], []
    dmg = {h: [] for h in SCAN}                       # per example: m_clean - m_clean|h->corrupt
    for i in range(0, len(X), args.batch):
        ch = X[i:i + args.batch]
        cl, co, dn = task.pair(ch)
        hold = {}
        corrupt_heads = capture_head_outputs(ad, lambda: hold.__setitem__("o", ad.model(**co)))
        mk = task.margin(hold["o"].logits.float(), dn)
        mc = task.margin(ad.model(**cl).logits.float(), dn)
        m_cl += mc.tolist(); m_co += mk.tolist()
        for h in SCAN:
            with patch_heads(ad, corrupt_heads, [h]):
                mn = task.margin(ad.model(**cl).logits.float(), dn)
            dmg[h] += (mc - mn).tolist()
        print(f"  scan {min(i + args.batch, len(X))}/{len(X)}", end="\r", flush=True)
    print()

    m_cl = np.asarray(m_cl); m_co = np.asarray(m_co)
    coup = m_cl - m_co
    scan = {}
    for h in SCAN:
        d = np.asarray(dmg[h])
        lo, hi = gboot_ratio(d, coup, genes)
        scan[hkey(h)] = {"layer": h[0], "head": h[1],
                         "damage": ratio_of_means(d, coup), "ci": [lo, hi]}

    # sanity gate
    d_reader = scan[hkey(READER)]["damage"]
    d_addr = scan[hkey(ADDRESSER)]["damage"]
    sane = (d_reader == max(scan[k]["damage"] for k in scan)) and d_reader > d_addr > 0.0

    # F-Heads: L4/L5 heads with CI-low > 0.05, ranked
    l45_ranked = sorted(
        (scan[hkey(h)] | {"key": hkey(h), "hh": h} for h in L45),
        key=lambda r: r["damage"], reverse=True)
    relays = [r for r in l45_ranked if r["ci"][0] > 0.05]
    f_heads = len(relays) > 0

    # ---- Part 2: router vs transporter on the top L4/L5 heads ---------------------------------
    top = [r["hh"] for r in l45_ranked[:TOP_K]]
    qk_v = {}
    for i in range(0, len(X), args.batch):
        ch = X[i:i + args.batch]
        cl, co, dn = task.pair(ch)
        S = cl["input_ids"].shape[1]
        allpos = torch.arange(S, device=ad.device)[None].expand(len(ch), S)
        cqkv = {l: capture_proj(ad, l, ["query", "key", "value"], lambda: ad.model(**co))
                for l in {h[0] for h in top}}
        mc = task.margin(ad.model(**cl).logits.float(), dn)
        for h in top:
            l, idx = h
            with patch_proj(ad, l, idx, "query", cqkv[l]["query"], allpos), \
                 patch_proj(ad, l, idx, "key", cqkv[l]["key"], allpos):
                m_qk = task.margin(ad.model(**cl).logits.float(), dn)
            with patch_proj(ad, l, idx, "value", cqkv[l]["value"], allpos):
                m_v = task.margin(ad.model(**cl).logits.float(), dn)
            qk_v.setdefault(hkey(h), {"qk": [], "v": []})
            qk_v[hkey(h)]["qk"] += (mc - m_qk).tolist()
            qk_v[hkey(h)]["v"] += (mc - m_v).tolist()
        print(f"  qkov {min(i + args.batch, len(X))}/{len(X)}", end="\r", flush=True)
    print()

    router = {}
    for h in top:
        k = hkey(h)
        dqk = ratio_of_means(np.asarray(qk_v[k]["qk"]), coup)
        dv = ratio_of_means(np.asarray(qk_v[k]["v"]), coup)
        cls_ = "router" if dqk - dv >= 0.05 else "transporter" if dv - dqk >= 0.05 else "mixed"
        router[k] = {"damage_QK": dqk, "damage_V": dv, "class": cls_}

    # ---- mean-ablation health: DESCRIPTIVE (are relays dedicated or general-purpose heads?) ----
    health = {}
    for r in relays[:TOP_K]:
        health[r["key"]] = task.mlm_health(X[:40], ctx=ablate(ad, [r["hh"]], mode="mean"))

    # F-Selective: noising is head-specific (a global-breakage artifact would flag ~every head)
    n_relays = len(relays)
    f_selective = n_relays <= len(L45) // 3
    verdict = {"sanity": bool(sane), "F_heads": bool(f_heads), "F_selective": bool(f_selective)}
    verdict["overall"] = all(verdict.values())

    R = {
        "split": args.split, "n": len(X), "device": dev,
        "coupling_mean": float(coup.mean()),
        "sanity": {"L3H1_damage": d_reader, "L2H15_damage": d_addr, "L1H12_damage": scan[hkey(L1H12)]["damage"]},
        "scan": scan,
        "l45_ranked": [{"key": r["key"], "damage": r["damage"], "ci": r["ci"]} for r in l45_ranked],
        "relays": [r["key"] for r in relays],
        "n_relays": n_relays, "n_l45_scanned": len(L45),
        "router_transporter": router,
        "health_mean_ablation_descriptive": health,
        "verdict": {k: bool(v) for k, v in verdict.items()},
        "REFLECTION": {"status": "PENDING -- fill after inspection (prereg section 6)"},
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=1))
    ExperimentManifest.create(
        "splicebert_coupling:donor_heads", sys.argv,
        config={"split": args.split, "n": len(X), "scanned": [list(h) for h in SCAN],
                "top_k": TOP_K, "seed": SEED},
        notes=["1c: resolves L4/L5 attention into heads by NOISING-to-corrupt (not ablation).",
               "Part 2 splits query+key (routing) vs value (content) on the top heads.",
               "Health uses mean-ablation as a stricter breakage control than noising."],
    ).write(out.parent / "donor_heads.manifest.json")

    print("\n" + "=" * 78)
    print(f"coupling {coup.mean():+.3f}   SANITY L3H1 {d_reader:.3f} > L2H15 {d_addr:.3f} "
          f"> L1H12 {scan[hkey(L1H12)]['damage']:.3f}   sane={sane}")
    print("L4/L5 heads by coupling-damage (noising), top 8:")
    for r in l45_ranked[:8]:
        print(f"    {r['key']:7s} {r['damage']:+.3f}  [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]"
              f"{'  <-- relay' if r['ci'][0] > 0.05 else ''}")
    print("router vs transporter (top heads):")
    for k, v in router.items():
        print(f"    {k:7s} QK {v['damage_QK']:+.3f}  V {v['damage_V']:+.3f}  -> {v['class']}")
    print(f"selectivity: {n_relays}/{len(L45)} L4/L5 heads are relays (CI-low>0.05)")
    print(f"mean-ablation health (descriptive, general-purpose if <0.95): {health}")
    print("\nVERDICT:")
    for k, v in verdict.items():
        print(f"  {k:9s} {'PASS' if v else 'FAIL'}")
    print("=" * 78)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
