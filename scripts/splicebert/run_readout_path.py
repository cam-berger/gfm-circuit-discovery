#!/usr/bin/env python
"""HOW DOES L3H1's OUTPUT REACH THE READOUT?  (the missing downstream edge)

We know L3H1 carries the coupling.  We do not know how its write becomes a donor logit.
"The head matters" is compatible with two very different circuits:

    DIRECT     L3H1 -> residual -> (LayerNorm) -> MLM head
    MEDIATED   L3H1 -> residual -> some L4/L5 head or MLP -> ... -> MLM head

PATH PATCHING.  Patch L3H1's clean output into the corrupt run, and simultaneously CLAMP
every downstream residual write -- MLP3, L4-attn, MLP4, L5-attn, MLP5 -- to the value it
took in the *unpatched corrupt* run.  Those blocks then contribute exactly what they would
have contributed with no patch at all, so they cannot propagate the patch.  Whatever
recovery survives reached the logit through the residual stream alone: the DIRECT effect.
Un-clamping one block at a time then prices each mediated route.

A SHARP PREDICTION, worth stating before the run.  The existing decomposition says patching
L3H1 at the DONOR position alone recovers 0.418, while patching it at ALL positions recovers
0.819.  The donor is the readout position, so its write should be almost entirely direct.
The other half is written at positions the MLM head never reads, so it can only matter by
being *attended to later* -- it should be almost entirely mediated.  If both halves come
back direct, the decomposition is broken, not the model.

DONOR-QUERY NECESSITY (D8).  A clean-query patch recovering 0.003 shows only that acceptor
corruption did not perturb the donor's query -- it says nothing about whether the query is
needed.  The necessity test is destructive: transplant L3H1's query at the donor from a
distance-matched non-site, in BOTH the clean and corrupt runs, and ask whether the coupling
survives a head that is asking the wrong question.
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

from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, READER, CouplingTask, build_windows,
    capture_head_outputs, capture_proj, capture_writes, clamp_write, downstream_blocks,
    gboot_ratio, load_adapter, load_examples, patch_heads, patch_proj, pick_device,
    ratio_of_means,
)

L3, H3 = READER
BLOCK_LABEL = {(3, "mlp"): "MLP3", (4, "attn"): "L4-attn", (4, "mlp"): "MLP4",
               (5, "attn"): "L5-attn", (5, "mlp"): "MLP5"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="results/splicebert_coupling/readout_path.json")
    args = ap.parse_args()

    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=True)   # attention is read below
    task = CouplingTask(ad, batch=args.batch)
    OFF = task.off
    DOWN = downstream_blocks(ad, L3)                 # [(3,mlp),(4,attn),(4,mlp),(5,attn),(5,mlp)]

    X, _ = load_examples(args.gtf, args.fasta, args.split, args.n, task=task,
                         tag=f"readout-{args.split}", min_effect=0.3)
    genes = [x.gene_id for x in X]
    print(f"{len(X)} {args.split} exons with a usable coupling denominator   device={dev}")
    print(f"downstream blocks clamped: {[BLOCK_LABEL[b] for b in DOWN]}")

    # conditions: (patch positions, which downstream blocks stay FROZEN)
    CONDS = [
        ("all-pos, downstream free",     "all",  []),
        ("all-pos, downstream FROZEN",   "all",  DOWN),
        ("donor-only, downstream free",  "don",  []),
        ("donor-only, downstream FROZEN", "don", DOWN),
        ("other-pos, downstream free",   "oth",  []),
        ("other-pos, downstream FROZEN", "oth",  DOWN),
    ]
    # mediator scan: freeze everything downstream EXCEPT one block
    for b in DOWN:
        CONDS.append((f"all-pos, only {BLOCK_LABEL[b]} live", "all", [x for x in DOWN if x != b]))
    # CUMULATIVE scan: un-freeze the stack from the bottom up.  Single-block un-freezing
    # cannot distinguish "no block mediates" from "the route is a CHAIN and every link is
    # required".  In a residual network the second is the common case, so measure it.
    for j in range(1, len(DOWN)):
        live = DOWN[:j]
        CONDS.append((f"all-pos, live<= {BLOCK_LABEL[DOWN[j-1]]}", "all",
                      [x for x in DOWN if x not in live]))

    num = {c[0]: [] for c in CONDS}
    dnm: list[float] = []
    prov = None

    with torch.no_grad():
        for i in range(0, len(X), args.batch):
            ch = X[i:i + args.batch]
            cl, co, dn = task.pair(ch)
            W = build_windows(ch, OFF, ad.device)
            prov = prov or W.provenance
            S = cl["input_ids"].shape[1]

            clean_heads = capture_head_outputs(ad, lambda: ad.model(**cl))
            corrupt_writes = capture_writes(ad, DOWN, lambda: ad.model(**co))
            mc = task.margin(ad.model(**cl).logits.float(), dn)
            mk = task.margin(ad.model(**co).logits.float(), dn)
            dnm += (mc - mk).tolist()

            don_mask = torch.zeros(len(ch), S, dtype=torch.bool, device=ad.device)
            for j, x in enumerate(ch):
                don_mask[j, x.donor_pos + OFF] = True
            oth_mask = ~don_mask
            oth_mask &= cl["attention_mask"].bool()   # never patch padding

            masks = {"all": None, "don": don_mask, "oth": oth_mask}
            for name, where, frozen in CONDS:
                with ExitStack() as st:
                    st.enter_context(patch_heads(ad, clean_heads, [READER],
                                                 token_mask=masks[where]))
                    if frozen:
                        st.enter_context(clamp_write(ad, frozen, corrupt_writes))
                    m = task.margin(ad.model(**co).logits.float(), dn)
                num[name] += (m - mk).tolist()
            print(f"  path {min(i+args.batch, len(X))}/{len(X)}", end="\r", flush=True)
    print()

    R: dict = {"split": args.split, "n": len(X), "paths": {}}
    print("\n" + "=" * 82)
    print("PATH DECOMPOSITION -- fraction of the coupling restored by patching L3H1")
    print(f"  {'condition':34s} {'recovery':>9s} {'95% CI':>18s}")
    for name, _, _ in CONDS:
        r = ratio_of_means(num[name], dnm)
        lo, hi = gboot_ratio(num[name], dnm, genes)
        R["paths"][name] = dict(recovery=r, lo=lo, hi=hi)
        print(f"  {name:34s} {r:9.3f}  [{lo:+.3f},{hi:+.3f}]")

    direct = R["paths"]["all-pos, downstream FROZEN"]["recovery"]
    total = R["paths"]["all-pos, downstream free"]["recovery"]
    R["direct_fraction"] = float(direct / max(total, 1e-9))
    print(f"\n  DIRECT effect = {direct:.3f} of {total:.3f} total "
          f"=> {100*direct/max(total,1e-9):.0f}% of L3H1's contribution reaches the MLM")
    print("  head through the residual stream, with every downstream block frozen.")

    print("\n  MEDIATION by each downstream block ALONE (un-freeze exactly one):")
    med = {}
    for b in DOWN:
        k = f"all-pos, only {BLOCK_LABEL[b]} live"
        med[BLOCK_LABEL[b]] = float(R["paths"][k]["recovery"] - direct)
        print(f"    {BLOCK_LABEL[b]:10s} {med[BLOCK_LABEL[b]]:+.3f}")
    R["mediation_single"] = med

    print("\n  CUMULATIVE (un-freeze the stack bottom-up).  If no single block mediates but")
    print("  the cumulative curve climbs, the route is a CHAIN: every link is required and")
    print("  no link is sufficient -- which is what a residual network usually does.")
    cum = {"direct (all frozen)": direct}
    print(f"    {'direct (all frozen)':22s} {direct:6.3f}")
    for j in range(1, len(DOWN)):
        k = f"all-pos, live<= {BLOCK_LABEL[DOWN[j-1]]}"
        cum[k] = float(R["paths"][k]["recovery"])
        print(f"    {'+ ' + BLOCK_LABEL[DOWN[j-1]]:22s} {cum[k]:6.3f}")
    cum["all live"] = total
    print(f"    {'all live':22s} {total:6.3f}")
    R["mediation_cumulative"] = cum

    # ---- D8: donor-query NECESSITY -------------------------------------------------------
    print("\n" + "=" * 82)
    print("D8  DONOR-QUERY NECESSITY  (transplant L3H1's query at the donor)")
    base_c, tq_c, mq_c = [], [], []
    att_base, att_tq = [], []
    clean_margin_base, clean_margin_tq = [], []
    with torch.no_grad():
        for i in range(0, len(X), args.batch):
            ch = X[i:i + args.batch]
            cl, co, dn = task.pair(ch)
            W = build_windows(ch, OFF, ad.device)
            don = W.donor                                   # [B,1]
            ctl = W.control[:, -1:]                          # [B,1] distance-matched non-site

            q_cl = capture_proj(ad, L3, ["query"], lambda: ad.model(**cl))["query"]
            q_co = capture_proj(ad, L3, ["query"], lambda: ad.model(**co))["query"]

            # source = the query the head would have formed at a NON-SITE, same run
            def transplanted(src):
                s = src.clone()
                b = torch.arange(s.shape[0], device=s.device)[:, None]
                s[b, don] = src[b, ctl]
                return s

            Dh = ad.hidden_size // ad.num_heads
            mean_q = q_cl[:, :, H3 * Dh:(H3 + 1) * Dh].mean(dim=(0, 1), keepdim=True)

            def meaned(src):
                s = src.clone()
                s[:, :, H3 * Dh:(H3 + 1) * Dh] = mean_q.expand_as(s[:, :, H3 * Dh:(H3 + 1) * Dh])
                return s

            o_cl = ad.model(**cl, output_attentions=True)
            m_cl = task.margin(o_cl.logits.float(), dn)
            m_co = task.margin(ad.model(**co).logits.float(), dn)
            base_c += (m_cl - m_co).tolist()
            clean_margin_base += m_cl.tolist()

            with patch_proj(ad, L3, H3, "query", transplanted(q_cl), don):
                o = ad.model(**cl, output_attentions=True)
                a_cl = task.margin(o.logits.float(), dn)
            with patch_proj(ad, L3, H3, "query", transplanted(q_co), don):
                a_co = task.margin(ad.model(**co).logits.float(), dn)
            tq_c += (a_cl - a_co).tolist()
            clean_margin_tq += a_cl.tolist()

            with patch_proj(ad, L3, H3, "query", meaned(q_cl), don):
                b_cl = task.margin(ad.model(**cl).logits.float(), dn)
            with patch_proj(ad, L3, H3, "query", meaned(q_co), don):
                b_co = task.margin(ad.model(**co).logits.float(), dn)
            mq_c += (b_cl - b_co).tolist()

            A0 = o_cl.attentions[L3][:, H3]
            A1 = o.attentions[L3][:, H3]
            for j, x in enumerate(ch):
                d = x.donor_pos + OFF
                p = x.next_acceptor_pos + OFF
                att_base.append(A0[j, d, p - 2:p + 1].sum().item())
                att_tq.append(A1[j, d, p - 2:p + 1].sum().item())

    base_c, tq_c, mq_c = map(np.asarray, (base_c, tq_c, mq_c))
    tlo, thi = gboot_ratio(base_c - tq_c, base_c, genes)
    mlo, mhi = gboot_ratio(base_c - mq_c, base_c, genes)
    print(f"  baseline coupling                       {base_c.mean():+.3f}")
    print(f"  query transplanted from a non-site      {tq_c.mean():+.3f}   "
          f"({100*ratio_of_means(base_c-tq_c, base_c):.0f}% lost [{100*tlo:.0f},{100*thi:.0f}])")
    print(f"  query mean-ablated                      {mq_c.mean():+.3f}   "
          f"({100*ratio_of_means(base_c-mq_c, base_c):.0f}% lost [{100*mlo:.0f},{100*mhi:.0f}])")
    print(f"  L3H1 attention donor->partner  intact {np.mean(att_base):.4f}"
          f"  -> transplanted query {np.mean(att_tq):.4f}")
    print(f"  clean donor margin             intact {np.mean(clean_margin_base):+.3f}"
          f"  -> transplanted query {np.mean(clean_margin_tq):+.3f}")
    print("  (the clean-margin column is the control: if it collapses too, the transplant")
    print("   broke the donor's representation rather than isolating the query's role.)")
    R["donor_query"] = dict(
        baseline=float(base_c.mean()),
        transplant=float(tq_c.mean()),
        transplant_frac_lost=ratio_of_means(base_c - tq_c, base_c), transplant_ci=[tlo, thi],
        mean_ablate=float(mq_c.mean()),
        mean_ablate_frac_lost=ratio_of_means(base_c - mq_c, base_c), mean_ablate_ci=[mlo, mhi],
        attention_intact=float(np.mean(att_base)),
        attention_transplant=float(np.mean(att_tq)),
        clean_margin_intact=float(np.mean(clean_margin_base)),
        clean_margin_transplant=float(np.mean(clean_margin_tq)),
    )
    R["patch_windows"] = prov

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=1))
    ExperimentManifest.create(
        "splicebert_coupling:readout_path", sys.argv,
        config={"split": args.split, "n": len(X), "reader": list(READER),
                "downstream_blocks": [list(b) for b in DOWN]},
        notes=["Clamped blocks are forced to their UNPATCHED-CORRUPT write, so they "
               "contribute what they would have with no patch and cannot propagate it.",
               "Layer-3 ATTENTION is never clamped -- that would overwrite the patch itself.",
               "D8 is a destructive test: the earlier 0.003 clean-query recovery was a null "
               "about the corruption, not about the query."],
    ).write(out.parent / "readout_path.manifest.json")
    print("=" * 82)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
