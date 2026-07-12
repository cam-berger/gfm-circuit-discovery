#!/usr/bin/env python
"""L2H15 x L3H1: factorial + directed path decomposition.

WHY.  The exhaustive held-out ablation scan found TWO substantial components:
    L3H1   drop_partner 0.944   (also denoise-recovers 0.808 -- necessary AND sufficient)
    L2H15  drop_partner 0.542   (INVISIBLE to denoising -- necessary, not sufficient)
Their marginal drops sum to 1.486 > the 1.121 baseline coupling, so they CANNOT be
independent additive shares.  They overlap.  The high-necessity / low-sufficiency
signature of L2H15 is the classic fingerprint of an UPSTREAM FEEDER, not a downstream
relay -- so the patches are scoped for an L2H15 -> L3H1 route.

PART 1 -- 2x2 ABLATION FACTORIAL
    drop_both ~ drop_L3            => L2H15 is nested inside L3H1 (upstream feeder)
    drop_both > drop_L3            => L2H15 also has a route that BYPASSES L3H1
    drop_both ~ drop_L3 + drop_L2  => closer to parallel contributors

PART 2 -- DIRECTED PATH PATCHING (denoise into the corrupt run; recovery of the coupling)
    R_L2_total    patch clean L2H15, everything downstream recomputes
    R_L2_acceptor patch clean L2H15 ONLY at the acceptor position (is that where it acts?)
    R_L2_cut      patch clean L2H15 *while CLAMPING L3H1 to its CORRUPT value*
                  -> severs the L2->L3 route; whatever survives is L2's BYPASS effect
    R_L2_via_L3 = R_L2_total - R_L2_cut          <- the mediated share
    R_L3_total    patch clean L3H1 (L2H15 stays corrupt) -- does L3H1 bypass the L2 lesion?
    R_both        patch both clean

Positional controls: L2H15 patched at the DONOR instead of the acceptor, and L3H1 patched
at the ACCEPTOR instead of the donor.  If the route is real it should be position-specific.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.adapters.splicebert import SpliceBertAdapter  # noqa: E402
from gfmcd.circuits.hf_bert import (  # noqa: E402
    capture_head_outputs, edit_heads, patch_heads,
)
from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.coupling import (  # noqa: E402
    ACCEPTOR_ALTS, CouplingConfig, build_coupling_dataset, stable_rng, substitute,
)
from gfmcd.splicing.dataset import PyfaidxGenome  # noqa: E402

L3, H3 = 3, 1       # necessary AND sufficient
L2, H2 = 2, 15      # necessary, NOT sufficient  -> suspected upstream feeder


@contextmanager
def _noop():
    yield


def gboot(v, g, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    by = {}
    for a, b in zip(v, g):
        if not np.isnan(a):
            by.setdefault(b, []).append(a)
    k = list(by)
    if not k:
        return (np.nan, np.nan)
    m = np.empty(n)
    for i in range(n):
        p = rng.choice(len(k), len(k), replace=True)
        m[i] = np.mean([x for j in p for x in by[k[j]]])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default="/Users/cberg/Research/data/genomes/gencode.v50.basic.annotation.gtf.gz")
    ap.add_argument("--fasta", default="/Users/cberg/Research/data/genomes/hg38.fa")
    ap.add_argument("--model", default="data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt")
    ap.add_argument("--out", default="results/splicebert_coupling/path_decomposition.json")
    args = ap.parse_args()

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    dev = args.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    mp = args.model if Path(args.model).is_absolute() else str(REPO / args.model)
    ad = SpliceBertAdapter(AutoModelForMaskedLM.from_pretrained(mp),
                           AutoTokenizer.from_pretrained(mp), device=dev)
    tok, MASK = ad.tokenizer, ad.tokenizer.mask_token_id
    BID = {b: tok.convert_tokens_to_ids(b) for b in "ACGT"}
    OFF = ad.prefix_tokens

    cfg = CouplingConfig()
    X = build_coupling_dataset(args.gtf, PyfaidxGenome(args.fasta), cfg)
    X = [x for x in X if x.split == args.split and x.next_acceptor_pos is not None
         and x.sequence[x.donor_pos:x.donor_pos + 2] == "GT"
         and 60 <= (x.next_acceptor_pos - x.donor_pos) <= 150]
    r = stable_rng("path", cfg.seed)
    r.shuffle(X)
    print(f"{len(X)} held-out exons ({args.split})  device={dev}   "
          f"L{L2}H{H2} (feeder?) x L{L3}H{H3}")

    def enc_of(seqs, dn):
        e = tok([" ".join(s) for s in seqs], return_tensors="pt", padding=True).to(ad.device)
        for j, d in enumerate(dn):
            e["input_ids"][j, d + OFF] = MASK
        return e

    def metric(lg, dn):
        i = torch.arange(len(dn), device=lg.device)
        p = torch.tensor([d + OFF for d in dn], device=lg.device)
        v = lg[i, p]
        return v[:, BID["G"]] - torch.stack([v[:, BID[b]] for b in "ACT"], -1).max(-1).values

    # PRE-REGISTERED FILTER: keep only exons where the coupling actually exists.
    # Without it, examples with ~zero (m_clean - m_corrupt) sit in the recovery
    # denominator, get clamped, and blow the ratio up (this produced recoveries of -14).
    keep = []
    with torch.no_grad():
        for i in range(0, min(len(X), args.n * 5), args.batch):
            ch = X[i:i + args.batch]
            dn = [x.donor_pos for x in ch]
            cl = enc_of([x.sequence for x in ch], dn)
            co = enc_of([substitute(x.sequence, x.next_acceptor_pos - 2, ACCEPTOR_ALTS[0])
                         for x in ch], dn)
            e = (metric(ad.model(**cl).logits.float(), dn)
                 - metric(ad.model(**co).logits.float(), dn))
            keep += [x for j, x in enumerate(ch) if float(e[j]) >= 0.3]
            if len(keep) >= args.n:
                break
    X = keep[: args.n]
    genes = [x.gene_id for x in X]
    print(f"  -> {len(X)} exons with a real coupling effect (>= 0.3)")

    # ---------- PART 1: 2x2 ABLATION FACTORIAL ----------------------------------
    def coupling(ctx):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), args.batch):
                ch = X[i:i + args.batch]
                dn = [x.donor_pos for x in ch]
                cl = enc_of([x.sequence for x in ch], dn)
                co = enc_of([substitute(x.sequence, x.next_acceptor_pos - 2, ACCEPTOR_ALTS[0])
                             for x in ch], dn)
                with ctx():
                    out += (metric(ad.model(**cl).logits.float(), dn)
                            - metric(ad.model(**co).logits.float(), dn)).tolist()
        return np.asarray(out)

    base = coupling(_noop)
    c_l2 = coupling(lambda: edit_heads(ad, [(L2, H2)], mode="mean"))
    c_l3 = coupling(lambda: edit_heads(ad, [(L3, H3)], mode="mean"))
    c_bt = coupling(lambda: edit_heads(ad, [(L2, H2), (L3, H3)], mode="mean"))
    d2, d3, dbt = (base - c_l2), (base - c_l3), (base - c_bt)

    print("\n" + "=" * 76)
    print("PART 1  2x2 ABLATION FACTORIAL  (coupling, held-out)")
    print(f"  baseline coupling            {base.mean():+.3f}  "
          f"[{gboot(base,genes)[0]:+.3f},{gboot(base,genes)[1]:+.3f}]")
    for lbl, dd in [(f"drop  L{L2}H{H2} ablated", d2), (f"drop  L{L3}H{H3} ablated", d3),
                    ("drop  BOTH ablated", dbt)]:
        lo, hi = gboot(dd, genes)
        print(f"  {lbl:28s} {dd.mean():+.3f}  [{lo:+.3f},{hi:+.3f}]")
    extra = dbt.mean() - d3.mean()
    lo, hi = gboot(dbt - d3, genes)
    print(f"\n  EXTRA drop from also ablating L{L2}H{H2}, given L{L3}H{H3} is already gone:")
    print(f"     drop_both - drop_L3 = {extra:+.3f}  [{lo:+.3f},{hi:+.3f}]")
    print(f"     (L{L2}H{H2}'s MARGINAL drop alone was {d2.mean():+.3f})")
    frac = extra / max(d2.mean(), 1e-9)
    print(f"     => {100*(1-frac):.0f}% of L{L2}H{H2}'s effect is NESTED inside L{L3}H{H3}")
    if hi < 0.1 * d2.mean():
        print("     => consistent with L2H15 being an UPSTREAM FEEDER of L3H1")
    elif lo > 0.7 * d2.mean():
        print("     => consistent with PARALLEL contributors")
    else:
        print("     => MIXED: partly mediated, partly bypassing")

    # ---------- PART 2: DIRECTED PATH PATCHING ---------------------------------
    def recover(patch_specs, clamp_l3=False):
        """patch_specs: list of (layer, head, scope) with scope in {'all','acc','don'}."""
        num, dnm = [], []
        with torch.no_grad():
            for i in range(0, len(X), args.batch):
                ch = X[i:i + args.batch]
                dn = [x.donor_pos for x in ch]
                cl = enc_of([x.sequence for x in ch], dn)
                co = enc_of([substitute(x.sequence, x.next_acceptor_pos - 2, ACCEPTOR_ALTS[0])
                             for x in ch], dn)
                clean_cache = capture_head_outputs(ad, lambda: ad.model(**cl))
                corr_cache = capture_head_outputs(ad, lambda: ad.model(**co))
                m_c = metric(ad.model(**cl).logits.float(), dn)
                m_k = metric(ad.model(**co).logits.float(), dn)
                den = (m_c - m_k).clamp_min(1e-3)

                S = cl["input_ids"].shape[1]
                masks = {}
                for scope in ("acc", "don"):
                    mm = torch.zeros((len(ch), S), dtype=torch.bool, device=ad.device)
                    for j, x in enumerate(ch):
                        p = (x.next_acceptor_pos if scope == "acc" else x.donor_pos) + OFF
                        mm[j, p] = True
                    masks[scope] = mm

                with ExitStack() as st:
                    for (l, h, scope) in patch_specs:
                        st.enter_context(patch_heads(
                            ad, clean_cache, [(l, h)],
                            token_mask=None if scope == "all" else masks[scope]))
                    if clamp_l3:
                        # freeze L3H1 at its CORRUPT value -> severs the L2 -> L3 route
                        st.enter_context(patch_heads(ad, corr_cache, [(L3, H3)]))
                    m = metric(ad.model(**co).logits.float(), dn)
                num += (m - m_k).tolist()
                dnm += (m_c - m_k).tolist()
        # RATIO OF MEANS -- robust.  A mean of per-example ratios is dominated by
        # examples whose denominator is near zero.
        num, dnm = np.asarray(num), np.asarray(dnm)
        return num, dnm

    print("\n" + "=" * 76)
    print("PART 2  DIRECTED PATH PATCHING  (fraction of the coupling recovered)")
    conds = {
        f"L{L3}H{H3} clean, all positions":            ([(L3, H3, "all")], False),
        f"L{L3}H{H3} clean, at DONOR only":            ([(L3, H3, "don")], False),
        f"L{L3}H{H3} clean, at ACCEPTOR only (ctrl)":  ([(L3, H3, "acc")], False),
        f"L{L2}H{H2} clean, all positions":            ([(L2, H2, "all")], False),
        f"L{L2}H{H2} clean, at ACCEPTOR only":         ([(L2, H2, "acc")], False),
        f"L{L2}H{H2} clean, at DONOR only (ctrl)":     ([(L2, H2, "don")], False),
        f"L{L2}H{H2} clean  ||  L{L3}H{H3} CLAMPED to corrupt": ([(L2, H2, "all")], True),
        f"BOTH clean":                                  ([(L2, H2, "all"), (L3, H3, "all")], False),
    }
    R = {}
    for lbl, (specs, clamp) in conds.items():
        num, dnm = recover(specs, clamp_l3=clamp)
        val = float(num.sum() / dnm.sum())
        # gene-grouped bootstrap on the RATIO OF MEANS
        rng = np.random.default_rng(0)
        by = {}
        for a, b, g in zip(num, dnm, genes):
            by.setdefault(g, []).append((a, b))
        gk = list(by)
        bs = np.empty(1000)
        for i in range(1000):
            pk = rng.choice(len(gk), len(gk), replace=True)
            pairs = [t for j in pk for t in by[gk[j]]]
            bs[i] = sum(t[0] for t in pairs) / max(sum(t[1] for t in pairs), 1e-9)
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        R[lbl] = dict(mean=val, lo=lo, hi=hi)
        print(f"  {lbl:48s} {val:6.3f}  [{lo:+.3f},{hi:+.3f}]")

    tot = R[f"L{L2}H{H2} clean, all positions"]["mean"]
    cut = R[f"L{L2}H{H2} clean  ||  L{L3}H{H3} CLAMPED to corrupt"]["mean"]
    print(f"\n  L{L2}H{H2} total recovery          = {tot:+.3f}")
    print(f"  L{L2}H{H2} with the L2->L3 route CUT = {cut:+.3f}   (bypass effect)")
    print(f"  => mediated via L{L3}H{H3}          = {tot-cut:+.3f} "
          f"({100*(tot-cut)/max(abs(tot),1e-9):.0f}% of L2's effect)")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "split": args.split, "n": len(X),
        "factorial": {"baseline": float(base.mean()), "drop_L2H15": float(d2.mean()),
                      "drop_L3H1": float(d3.mean()), "drop_both": float(dbt.mean()),
                      "extra_over_L3": float(extra),
                      "extra_ci": list(gboot(dbt - d3, genes))},
        "paths": R,
    }, indent=1))
    ExperimentManifest.create(
        "splicebert_coupling:path_decomposition", sys.argv,
        config={"n": len(X), "split": args.split, "heads": [[L2, H2], [L3, H3]]},
        notes=["Marginal ablation drops are NOT additive shares: 0.944+0.542 > baseline 1.121.",
               "Path cut = patch clean L2H15 while clamping L3H1 to its corrupt value."],
    ).write(out.parent / "path_decomposition.manifest.json")
    print("=" * 76)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
