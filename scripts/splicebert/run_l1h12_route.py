#!/usr/bin/env python
"""WHERE does L1H12 enter L3H1?  (the same instrument that cracked L2H15)

What we know.  L1H12 is necessary (18-26% of the coupling), carries NO partner signal
(denoise recovery 0.003), never looks at the partner (attention 0.003 vs L3H1's 0.62), is 95%
NESTED inside L3H1 -- and is NOT nested inside L2H15 (it adds +0.284 [0.221, 0.348] on top of
it).  So it is a SECOND, PARALLEL feeder into the reader.

Its period-3 attention comb turned out to be a red herring FOR THIS HEAD: it is purely
positional (survives sequence shuffling), it appears from any query position, and 46 of 96
heads have one.  It is a global property of SpliceBERT's learned position code, not L1H12's
job description.

THE HYPOTHESIS.  L2H15 feeds L3H1's KEY -- the ADDRESS, at the acceptor (rescue 0.43 via
key@acceptor, and value-only rescue is ~0).  If L1H12 is a parallel feeder, the obvious place
for it to enter is the other side of the same read: L3H1's QUERY, at the donor.  That would
make the circuit

        L2H15 ──▶ the ADDRESS  (key @ acceptor)   ┐
                                                  ├──▶ L3H1's read
        L1H12 ──▶ the QUESTION (query @ donor)    ┘

and it would explain why the two are independent of each other but both nested in L3H1.

THE TEST.  Ablate L1H12; then restore one piece of L3H1's input to its L1H12-INTACT value and
see how much of the lost coupling comes back.

    rescue(Z) = [coupling(L1H12 ablated, Z restored) - coupling(L1H12 ablated)]
                / [coupling(baseline)                - coupling(L1H12 ablated)]

The final row restores the ENTIRE layer-1 output and must return 1.000 BY CONSTRUCTION.  It is
the instrument's calibration, not a result: if it comes back as anything else, the rescue
arithmetic is wrong and no other row may be believed.

The hypothesis is falsifiable: if query@donor rescues ~0 and key@acceptor rescues a lot, then
L1H12 is a second ADDRESS-builder and the story above is wrong.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.circuits.hf_bert_residual import capture_layer_outputs, patch_layer_output  # noqa: E402
from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    ADDRESSER, DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, READER, CouplingTask, ablate,
    build_windows, capture_head_outputs, capture_proj, capture_writes, clamp_write,
    gboot_ratio, load_adapter, load_examples, patch_heads, patch_proj, pick_device,
    ratio_of_means,
)

L3, H3 = READER
L2, H2 = ADDRESSER
CAND = (1, 12)
L1, H1 = CAND

CONDS = [
    "L3H1 query @donor",            # THE HYPOTHESIS
    "L3H1 query @all-pos",
    "L3H1 key @acceptor",           # L2H15's route -- does L1H12 use it too?
    "L3H1 key @all-pos",
    "L3H1 value @all-pos",
    "L3H1 key+value @all-pos",
    "L3H1 head output @all-pos",    # everything L3H1 could do differently
    "L2H15 head output",            # does L1H12 route THROUGH L2H15?
    "MLP1 write @all-pos",
    "layer-1 output (CALIBRATION)",  # must be 1.000
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="results/splicebert_coupling/l1h12_route.json")
    args = ap.parse_args()

    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev)
    task = CouplingTask(ad, batch=args.batch)
    OFF = task.off
    ABL = ablate(ad, [CAND])
    MLP1 = [(L1, "mlp")]

    X, _ = load_examples(args.gtf, args.fasta, args.split, args.n, task=task,
                         tag=f"l1route-{args.split}", min_effect=0.3)
    genes = [x.gene_id for x in X]
    print(f"{len(X)} {args.split} exons   device={dev}   ablating L{L1}H{H1}\n")

    num = {c: [] for c in CONDS}
    base_l, abl_l, dnm = [], [], []
    prov = None

    with torch.no_grad():
        for i in range(0, len(X), task.batch):
            ch = X[i:i + task.batch]
            cl, co, dn = task.pair(ch)
            W = build_windows(ch, OFF, ad.device)
            prov = prov or W.provenance
            acc, don = W.acceptor, W.donor
            S = cl["input_ids"].shape[1]
            allp = torch.arange(S, device=ad.device)[None].expand(len(ch), -1)

            def sources(enc):
                proj = capture_proj(ad, L3, ["query", "key", "value"], lambda: ad.model(**enc))
                heads = capture_head_outputs(ad, lambda: ad.model(**enc))
                writes = capture_writes(ad, MLP1, lambda: ad.model(**enc))
                layers = capture_layer_outputs(ad, lambda: ad.model(**enc))
                return proj, heads, writes, layers

            p_cl, h_cl, w_cl, y_cl = sources(cl)
            p_co, h_co, w_co, y_co = sources(co)

            m_cl = task.margin(ad.model(**cl).logits.float(), dn)
            m_co = task.margin(ad.model(**co).logits.float(), dn)
            cbase = m_cl - m_co
            with ABL():
                a_cl = task.margin(ad.model(**cl).logits.float(), dn)
                a_co = task.margin(ad.model(**co).logits.float(), dn)
            cabl = a_cl - a_co

            base_l += cbase.tolist()
            abl_l += cabl.tolist()
            dnm += (cbase - cabl).tolist()

            def restore(proj, heads, writes, layers, which):
                st = ExitStack()
                st.enter_context(ABL())
                if which == "L3H1 query @donor":
                    st.enter_context(patch_proj(ad, L3, H3, "query", proj["query"], don))
                elif which == "L3H1 query @all-pos":
                    st.enter_context(patch_proj(ad, L3, H3, "query", proj["query"], allp))
                elif which == "L3H1 key @acceptor":
                    st.enter_context(patch_proj(ad, L3, H3, "key", proj["key"], acc))
                elif which == "L3H1 key @all-pos":
                    st.enter_context(patch_proj(ad, L3, H3, "key", proj["key"], allp))
                elif which == "L3H1 value @all-pos":
                    st.enter_context(patch_proj(ad, L3, H3, "value", proj["value"], allp))
                elif which == "L3H1 key+value @all-pos":
                    st.enter_context(patch_proj(ad, L3, H3, "key", proj["key"], allp))
                    st.enter_context(patch_proj(ad, L3, H3, "value", proj["value"], allp))
                elif which == "L3H1 head output @all-pos":
                    st.enter_context(patch_heads(ad, heads, [READER]))
                elif which == "L2H15 head output":
                    st.enter_context(patch_heads(ad, heads, [ADDRESSER]))
                elif which == "MLP1 write @all-pos":
                    st.enter_context(clamp_write(ad, MLP1, writes))
                elif which == "layer-1 output (CALIBRATION)":
                    st.enter_context(patch_layer_output(ad, layers, L1))
                else:
                    raise ValueError(which)
                return st

            for c in CONDS:
                with restore(p_cl, h_cl, w_cl, y_cl, c):
                    r_cl = task.margin(ad.model(**cl).logits.float(), dn)
                with restore(p_co, h_co, w_co, y_co, c):
                    r_co = task.margin(ad.model(**co).logits.float(), dn)
                num[c] += ((r_cl - r_co) - cabl).tolist()
            print(f"  {min(i+task.batch, len(X))}/{len(X)}", end="\r", flush=True)
    print()

    base_l, abl_l, dnm = map(np.asarray, (base_l, abl_l, dnm))
    print("\n" + "=" * 80)
    print(f"WHERE DOES L{L1}H{H1} ENTER L3H1?   ({args.split}, n={len(X)})")
    print(f"  baseline coupling            {base_l.mean():+.3f}")
    print(f"  with L1H12 ablated           {abl_l.mean():+.3f}"
          f"   (L1H12 supplies {dnm.mean():+.3f})")
    print(f"\n  {'restore (L1H12 still ablated)':30s} {'rescue':>8s} {'95% CI':>18s}")
    R: dict = {}
    for c in CONDS:
        v = ratio_of_means(num[c], dnm)
        lo, hi = gboot_ratio(num[c], dnm, genes)
        R[c] = dict(rescue=v, lo=lo, hi=hi)
        flag = "  <- MUST be 1.000" if "CALIBRATION" in c else ""
        print(f"  {c:30s} {v:8.3f}  [{lo:+.3f},{hi:+.3f}]{flag}")

    cal = R["layer-1 output (CALIBRATION)"]["rescue"]
    if abs(cal - 1.0) > 0.05:
        print(f"\n  *** CALIBRATION FAILED ({cal:.3f} != 1.000). The rescue arithmetic is wrong")
        print("      and no row above may be reported.")
    else:
        q = R["L3H1 query @donor"]["rescue"]
        k = R["L3H1 key @acceptor"]["rescue"]
        h = R["L3H1 head output @all-pos"]["rescue"]
        print(f"\n  calibration OK ({cal:.3f}).")
        print(f"  L3H1's whole output rescues {h:.3f} of L1H12's effect"
              f"  ({'a CLOSED edge' if h > 0.85 else 'NOT closed -- L1H12 also acts elsewhere'})")
        if q > 0.3 and q > 2 * k:
            print(f"  => L1H12 enters through the QUERY at the donor ({q:.3f}) far more than")
            print(f"     through the key at the acceptor ({k:.3f}).  It builds the QUESTION;")
            print("     L2H15 builds the ADDRESS.  Two independent feeders, one read.")
        elif k > 0.3 and k > 2 * q:
            print(f"  => L1H12 enters through the KEY at the acceptor ({k:.3f}), not the query")
            print(f"     ({q:.3f}).  HYPOTHESIS REFUTED: it is a second address-builder.")
        else:
            print(f"  => neither channel dominates (query {q:.3f}, key {k:.3f}).  L1H12's")
            print("     contribution is diffuse; do not force the parallel-feeder story.")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "split": args.split, "n": len(X), "head": list(CAND),
        "baseline_coupling": float(base_l.mean()),
        "coupling_ablated": float(abl_l.mean()),
        "coupling_supplied": float(dnm.mean()),
        "rescue": R, "calibration_ok": bool(abs(cal - 1.0) <= 0.05),
        "patch_windows": prov,
    }, indent=1))
    ExperimentManifest.create(
        "splicebert_coupling:l1h12_route", sys.argv,
        config={"split": args.split, "n": len(X), "candidate": list(CAND)},
        notes=["EXPLORATORY.  The hypothesis (L1H12 builds L3H1's QUERY) is falsifiable: a "
               "large key@acceptor rescue with a null query@donor rescue kills it.",
               "The layer-1-output row is calibration, not a result; it must equal 1.000."],
    ).write(out.parent / "l1h12_route.manifest.json")
    print("=" * 80)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
