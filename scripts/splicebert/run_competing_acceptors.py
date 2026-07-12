#!/usr/bin/env python
"""RETRACTED 2026-07-12 (R7) -- DO NOT RUN, DO NOT CITE.

This script SWAPS L3H1's (key, value) pair between the partner acceptor and a decoy.  That is
an ALGEBRAIC IDENTITY: transposing (k,v) pairs permutes the summation index of
out_i = sum_j softmax(q_i . k_j) v_j, so the head's output is unchanged EXACTLY, for any
weights.  Verified on a randomly-initialised model (|both-swap - baseline| = 1.2e-06).
It measured nothing, and the "address/content double dissociation" it produced is withdrawn.

Superseded by run_sealed_test.py arm D, which uses a ONE-WAY COPY (relocate the partner's
key/value to a decoy).  That changes the multiset of (k,v) pairs and can fail -- and does:
the partner's value with no address recovers 0.004.

Kept only so the retraction is auditable.  See studies/.../CLAIMS.md R7.
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
from gfmcd.circuits.guards import PatchWindow, acceptor_motif_positions, validate_all  # noqa: E402
from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.coupling import (  # noqa: E402
    ACCEPTOR_ALTS, CouplingConfig, build_coupling_dataset, stable_rng, substitute,
)
from gfmcd.splicing.dataset import PyfaidxGenome  # noqa: E402

L3, H3 = 3, 1
WIN = (-3, -2, -1, 0)          # the AG window: [a-2, a-1] plus one base either side


@contextmanager
def swap_proj(ad, layer, head, which, src, pos_a, pos_b):
    """Swap head `head`'s KEY or VALUE between two token windows.  pos_*: [B,K] indices."""
    mod = getattr(ad.attention_self_modules[layer], which)
    Dh = ad.model.config.hidden_size // ad.num_heads
    lo, hi = head * Dh, (head + 1) * Dh

    def hook(_m, _i, out):
        out = out.clone()
        b = torch.arange(out.shape[0], device=out.device)[:, None].expand_as(pos_a)
        a_vals = src[b, pos_a, lo:hi].clone()
        b_vals = src[b, pos_b, lo:hi].clone()
        out[b, pos_a, lo:hi] = b_vals      # the decoy's projection now sits at the partner
        out[b, pos_b, lo:hi] = a_vals
        return out
    h = mod.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


def gboot(v, g, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    by = {}
    for a, b in zip(v, g):
        by.setdefault(b, []).append(a)
    k = list(by)
    m = np.empty(n)
    for i in range(n):
        p = rng.choice(len(k), len(k), replace=True)
        m[i] = np.mean([x for j in p for x in by[k[j]]])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--dist-tol", type=int, default=40, help="decoy distance match, nt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default="/Users/cberg/Research/data/genomes/gencode.v50.basic.annotation.gtf.gz")
    ap.add_argument("--fasta", default="/Users/cberg/Research/data/genomes/hg38.fa")
    ap.add_argument("--model", default="data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt")
    ap.add_argument("--out", default="results/splicebert_coupling/competing_acceptors.json")
    args = ap.parse_args()

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    dev = args.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    mp = args.model if Path(args.model).is_absolute() else str(REPO / args.model)
    ad = SpliceBertAdapter(
        AutoModelForMaskedLM.from_pretrained(mp, attn_implementation="eager"),
        AutoTokenizer.from_pretrained(mp), device=dev)
    tok, MASK = ad.tokenizer, ad.tokenizer.mask_token_id
    BID = {b: tok.convert_tokens_to_ids(b) for b in "ACGT"}
    OFF = ad.prefix_tokens

    cfg = CouplingConfig()
    X = build_coupling_dataset(args.gtf, PyfaidxGenome(args.fasta), cfg)
    X = [x for x in X if x.split == "val" and x.next_acceptor_pos is not None
         and x.sequence[x.donor_pos:x.donor_pos + 2] == "GT"
         and 60 <= (x.next_acceptor_pos - x.donor_pos) <= 150]
    rr = stable_rng("compete", cfg.seed)
    rr.shuffle(X)

    # pick a DISTANCE-MATCHED decoy AG (never the partner, never adjacent to it)
    pairs = []
    for x in X:
        d, pt = x.donor_pos, x.next_acceptor_pos
        dist = pt - d
        cands = [p for p in x.other_ag
                 if p > d + 20 and abs((p - d) - dist) <= args.dist_tol
                 and abs(p - pt) > 12 and p + 1 < len(x.sequence) and p - 3 > 0]
        if cands:
            pd = min(cands, key=lambda p: abs((p - d) - dist))
            pairs.append((x, pt, pd))
        if len(pairs) >= args.n:
            break
    print(f"{len(pairs)} held-out exons with a distance-matched decoy AG "
          f"(|Δdistance| ≤ {args.dist_tol}nt)   device={dev}")

    def enc_of(seqs, dn):
        e = tok([" ".join(s) for s in seqs], return_tensors="pt", padding=True).to(ad.device)
        for j, dd in enumerate(dn):
            e["input_ids"][j, dd + OFF] = MASK
        return e

    def metric(lg, dn):
        i = torch.arange(len(dn), device=lg.device)
        p = torch.tensor([dd + OFF for dd in dn], device=lg.device)
        v = lg[i, p]
        return v[:, BID["G"]] - torch.stack([v[:, BID[b]] for b in "ACT"], -1).max(-1).values

    CONDS = ["baseline", "key-swap", "value-swap", "both-swap"]
    marg = {c: [] for c in CONDS}
    att_t = {c: [] for c in CONDS}
    att_d = {c: [] for c in CONDS}
    m_corrupt, genes = [], []
    prov = None

    with torch.no_grad():
        for i in range(0, len(pairs), args.batch):
            ch = pairs[i:i + args.batch]
            xs = [t[0] for t in ch]
            dn = [x.donor_pos for x in xs]
            genes += [x.gene_id for x in xs]
            cl = enc_of([x.sequence for x in xs], dn)
            co = enc_of([substitute(x.sequence, x.next_acceptor_pos - 2, ACCEPTOR_ALTS[0])
                         for x in xs], dn)
            m_corrupt += metric(ad.model(**co).logits.float(), dn).tolist()

            # source projections from the CLEAN run
            st = {}
            mods = {w: getattr(ad.attention_self_modules[L3], w) for w in ("key", "value")}
            hs = [m.register_forward_hook(
                lambda _m, _i, o, w=w: st.__setitem__(w, o.detach())) for w, m in mods.items()]
            try:
                ad.model(**cl)
            finally:
                for h in hs:
                    h.remove()

            PT = torch.tensor([[t[1] + OFF + k for k in WIN] for t in ch], device=ad.device)
            PD = torch.tensor([[t[2] + OFF + k for k in WIN] for t in ch], device=ad.device)
            if prov is None:
                corr0 = acceptor_motif_positions(ch[0][1], OFF)
                prov = validate_all([
                    PatchWindow("swap@partner", tuple(int(v) for v in PT[0]), corr0),
                    PatchWindow("swap@decoy", tuple(int(v) for v in PD[0]), corr0,
                                is_control=True),   # the decoy window must NOT touch the AG
                ])

            specs = {"baseline": [], "key-swap": ["key"], "value-swap": ["value"],
                     "both-swap": ["key", "value"]}
            for c, which in specs.items():
                with ExitStack() as stk:
                    for w in which:
                        stk.enter_context(swap_proj(ad, L3, H3, w, st[w], PT, PD))
                    out = ad.model(**cl, output_attentions=True)
                marg[c] += metric(out.logits.float(), dn).tolist()
                A = out.attentions[L3][:, H3]
                for j, (x, pt, pd) in enumerate(ch):
                    dd = x.donor_pos + OFF
                    att_t[c].append(A[j, dd, pt + OFF - 2:pt + OFF + 1].sum().item())
                    att_d[c].append(A[j, dd, pd + OFF - 2:pd + OFF + 1].sum().item())

    base = np.mean(marg["baseline"])
    corr = np.mean(m_corrupt)
    print("\n" + "=" * 84)
    print("COMPETING ACCEPTORS -- swap L3H1's projections between the partner and a decoy AG")
    print(f"  reference: clean margin {base:+.3f}   partner-destroyed margin {corr:+.3f}"
          f"   (span {base-corr:.3f})")
    print(f"\n  {'condition':12s} {'attn->PARTNER':>14s} {'attn->DECOY':>12s} "
          f"{'donor margin':>13s} {'% of partner effect lost':>25s}")
    R = {}
    for c in CONDS:
        at, adk, mg = np.mean(att_t[c]), np.mean(att_d[c]), np.mean(marg[c])
        lost = (base - mg) / max(base - corr, 1e-9)
        lo, hi = gboot(np.asarray(marg[c]), genes)
        R[c] = dict(attn_partner=float(at), attn_decoy=float(adk), margin=float(mg),
                    margin_ci=[lo, hi], frac_partner_effect_lost=float(lost))
        print(f"  {c:12s} {at:14.4f} {adk:12.4f} {mg:13.3f} {100*lost:24.0f}%")

    ks, vs = R["key-swap"], R["value-swap"]
    b = R["baseline"]
    print("\n  DOUBLE DISSOCIATION TEST")
    moved = ks["attn_partner"] < 0.5 * b["attn_partner"] and ks["attn_decoy"] > b["attn_decoy"]
    stayed = vs["attn_partner"] > 0.8 * b["attn_partner"]
    hurt = vs["frac_partner_effect_lost"] > 0.25
    print(f"    key-swap MOVES attention to the decoy   : {moved}  "
          f"({b['attn_partner']:.3f}->{ks['attn_partner']:.3f} on the partner, "
          f"{b['attn_decoy']:.3f}->{ks['attn_decoy']:.3f} on the decoy)")
    print(f"    value-swap LEAVES attention on the partner: {stayed}  "
          f"({b['attn_partner']:.3f}->{vs['attn_partner']:.3f})")
    print(f"    value-swap still DAMAGES the margin      : {hurt}  "
          f"({100*vs['frac_partner_effect_lost']:.0f}% of the partner effect lost)")
    if moved and stayed and hurt:
        print("\n    => ADDRESS and CONTENT are SEPARABLE.  The KEY selects WHICH acceptor is")
        print("       read; the VALUE determines WHAT is read from it.  L3H1 BINDS the donor")
        print("       to a PARTICULAR acceptor -- this is partner binding, not 'reads an AG'.")
    else:
        print("\n    => NOT established.  The channels are entangled, or the decoy is not a")
        print("       real competitor.  Do NOT claim binding.")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n": len(pairs), "clean_margin": float(base),
                               "corrupt_margin": float(corr), "conditions": R,
                               "patch_windows": prov}, indent=1))
    ExperimentManifest.create(
        "splicebert_coupling:competing_acceptors", sys.argv,
        config={"n": len(pairs), "dist_tol": args.dist_tol, "head": [L3, H3]},
        notes=["Within-sequence double dissociation: attention should follow the KEY, the "
               "conclusion should follow the VALUE."],
    ).write(out.parent / "competing_acceptors.manifest.json")
    print("=" * 84)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
