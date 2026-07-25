#!/usr/bin/env python
"""D11 -- WHERE DOES THE REST OF L2H15's EFFECT GO?

L2H15 supplies +0.661 nats of coupling (baseline 1.384 -> 0.723 ablated, ~48%; R29 -- 0.72 is the
coupling RETAINED after ablation, not the amount supplied).  Restoring L3H1's KEY at the acceptor rescues 0.42 of
that, and MLP2's write at the acceptor rescues 0.28.  Roughly 60% is unaccounted for, and
an unaccounted 60% is not a footnote -- it is the difference between "we have mapped this
edge" and "we have found one of its strands".

The rescue is nested, so the point at which it SATURATES localises the route:

    restore ... to its L2H15-INTACT value, while L2H15 stays ablated
      L3H1 key @ acceptor window        the published edge
      L3H1 key @ ALL positions          does the address matter away from the AG?
      L3H1 key+value @ ALL positions
      L3H1 query @ donor                does L2H15 also shape the QUESTION?
      L3H1 whole head output            everything L3H1 could possibly do differently
      ALL layer-3 head outputs          everything layer-3 ATTENTION could do differently
      MLP2 write @ acceptor / @ all     the L2H15 -> MLP2 -> L3H1 route
      layer-2 residual output           EVERYTHING L2H15 changed  ->  MUST rescue 1.000

That last row is not a result, it is the instrument's calibration: restoring the entire
layer-2 output while L2H15 is ablated makes every downstream computation identical to the
intact model, so its rescue is 1.000 BY CONSTRUCTION.  If it comes back as anything else,
the rescue arithmetic is wrong and no other row on this table may be believed.

    rescue(Z) = [ coupling(L2H15 ablated, Z restored) - coupling(L2H15 ablated) ]
                / [ coupling(baseline)                - coupling(L2H15 ablated) ]

Ratio of means, gene-grouped bootstrap.  Never mean-of-ratios.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="results/splicebert_coupling/l2_accounting.json")
    args = ap.parse_args()

    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev)
    task = CouplingTask(ad, batch=args.batch)
    OFF = task.off
    ABL = ablate(ad, [ADDRESSER])
    L3_HEADS = [(L3, h) for h in range(ad.num_heads)]
    MLP2 = [(L2, "mlp")]

    X, _ = load_examples(args.gtf, args.fasta, args.split, args.n, task=task,
                         tag=f"l2acct-{args.split}", min_effect=0.3)
    genes = [x.gene_id for x in X]
    print(f"{len(X)} {args.split} exons   device={dev}")

    CONDS = [
        "L3H1 key @acceptor", "L3H1 key @all-pos", "L3H1 key+value @all-pos",
        "L3H1 query @donor", "L3H1 key+value+query", "L3H1 head output @all-pos",
        "ALL layer-3 heads @all-pos", "MLP2 write @acceptor", "MLP2 write @all-pos",
        "layer-3 heads + MLP2", "layer-2 output (CALIBRATION)",
    ]
    num = {c: [] for c in CONDS}
    base_l, abl_l, dnm = [], [], []
    prov = None

    with torch.no_grad():
        for i in range(0, len(X), args.batch):
            ch = X[i:i + args.batch]
            cl, co, dn = task.pair(ch)
            W = build_windows(ch, OFF, ad.device)
            prov = prov or W.provenance
            acc, don = W.acceptor, W.donor

            # SOURCES: everything captured with L2H15 INTACT, separately for the clean and
            # the corrupt sequence.  The coupling is a difference of two runs, so each run
            # needs its own restore source.
            def sources(enc):
                proj = capture_proj(ad, L3, ["query", "key", "value"],
                                    lambda: ad.model(**enc))
                heads = capture_head_outputs(ad, lambda: ad.model(**enc))
                writes = capture_writes(ad, MLP2, lambda: ad.model(**enc))
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
            dnm += (cbase - cabl).tolist()            # the coupling L2H15 supplies

            ALLPOS = None                              # patch_proj needs indices, not a mask
            all_idx = torch.arange(cl["input_ids"].shape[1], device=ad.device)
            all_idx = all_idx[None].expand(len(ch), -1)

            def restore(proj, heads, writes, layers, enc, which):
                """Contexts that put `which` back to its L2H15-intact value."""
                st = ExitStack()
                st.enter_context(ABL())
                if which == "L3H1 key @acceptor":
                    st.enter_context(patch_proj(ad, L3, H3, "key", proj["key"], acc))
                elif which == "L3H1 key @all-pos":
                    st.enter_context(patch_proj(ad, L3, H3, "key", proj["key"], all_idx))
                elif which == "L3H1 key+value @all-pos":
                    st.enter_context(patch_proj(ad, L3, H3, "key", proj["key"], all_idx))
                    st.enter_context(patch_proj(ad, L3, H3, "value", proj["value"], all_idx))
                elif which == "L3H1 query @donor":
                    st.enter_context(patch_proj(ad, L3, H3, "query", proj["query"], don))
                elif which == "L3H1 key+value+query":
                    st.enter_context(patch_proj(ad, L3, H3, "key", proj["key"], all_idx))
                    st.enter_context(patch_proj(ad, L3, H3, "value", proj["value"], all_idx))
                    st.enter_context(patch_proj(ad, L3, H3, "query", proj["query"], all_idx))
                elif which == "L3H1 head output @all-pos":
                    st.enter_context(patch_heads(ad, heads, [READER]))
                elif which == "ALL layer-3 heads @all-pos":
                    st.enter_context(patch_heads(ad, heads, L3_HEADS))
                elif which == "MLP2 write @acceptor":
                    st.enter_context(clamp_write(ad, MLP2, writes, positions=acc))
                elif which == "MLP2 write @all-pos":
                    st.enter_context(clamp_write(ad, MLP2, writes))
                elif which == "layer-3 heads + MLP2":
                    st.enter_context(patch_heads(ad, heads, L3_HEADS))
                    st.enter_context(clamp_write(ad, MLP2, writes))
                elif which == "layer-2 output (CALIBRATION)":
                    st.enter_context(patch_layer_output(ad, layers, L2))
                else:
                    raise ValueError(which)
                return st

            for c in CONDS:
                with restore(p_cl, h_cl, w_cl, y_cl, cl, c):
                    r_cl = task.margin(ad.model(**cl).logits.float(), dn)
                with restore(p_co, h_co, w_co, y_co, co, c):
                    r_co = task.margin(ad.model(**co).logits.float(), dn)
                num[c] += ((r_cl - r_co) - cabl).tolist()
            print(f"  {min(i+args.batch, len(X))}/{len(X)}", end="\r", flush=True)
    print()

    base_l, abl_l, dnm = map(np.asarray, (base_l, abl_l, dnm))
    print("\n" + "=" * 80)
    print(f"L2H15 ACCOUNTING  ({args.split}, n={len(X)})")
    print(f"  baseline coupling             {base_l.mean():+.3f}")
    print(f"  with L2H15 ablated            {abl_l.mean():+.3f}"
          f"   (L2H15 supplies {dnm.mean():+.3f})")
    print(f"\n  {'restore (L2H15 still ablated)':32s} {'rescue':>8s} {'95% CI':>18s}")
    R: dict = {}
    for c in CONDS:
        v = ratio_of_means(num[c], dnm)
        lo, hi = gboot_ratio(num[c], dnm, genes)
        R[c] = dict(rescue=v, lo=lo, hi=hi)
        flag = "  <- MUST be 1.000" if "CALIBRATION" in c else ""
        print(f"  {c:32s} {v:8.3f}  [{lo:+.3f},{hi:+.3f}]{flag}")

    cal = R["layer-2 output (CALIBRATION)"]["rescue"]
    if abs(cal - 1.0) > 0.05:
        print(f"\n  *** CALIBRATION FAILED: {cal:.3f} != 1.000.  The rescue arithmetic is")
        print("      wrong and every other row above is uninterpretable.  Do not report them.")
    else:
        print(f"\n  calibration OK ({cal:.3f}) -- the rescue arithmetic is sound.")
        l3 = R["ALL layer-3 heads @all-pos"]["rescue"]
        h1 = R["L3H1 head output @all-pos"]["rescue"]
        print(f"  L3H1 alone carries {100*h1:.0f}% of L2H15's effect; ALL of layer-3 "
              f"attention carries {100*l3:.0f}%.")
        print(f"  => {100*(1-l3):.0f}% of L2H15's effect does NOT route through layer-3 "
              "attention at all.")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "split": args.split, "n": len(X),
        "baseline_coupling": float(base_l.mean()),
        "coupling_L2H15_ablated": float(abl_l.mean()),
        "coupling_supplied_by_L2H15": float(dnm.mean()),
        "rescue": R, "calibration_ok": bool(abs(cal - 1.0) <= 0.05),
        "patch_windows": prov,
    }, indent=1))
    ExperimentManifest.create(
        "splicebert_coupling:l2_accounting", sys.argv,
        config={"split": args.split, "n": len(X), "feeder": list(ADDRESSER),
                "reader": list(READER)},
        notes=["Nested rescue: the saturation point localises the route.",
               "The layer-2-output row is an instrument calibration and must equal 1.000; "
               "if it does not, no other row is interpretable."],
    ).write(out.parent / "l2_accounting.manifest.json")
    print("=" * 80)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
