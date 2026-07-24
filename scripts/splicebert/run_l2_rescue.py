#!/usr/bin/env python
"""L2H15 RESCUE + bounded MLP2 mediation.

QUESTION 1 -- is the L2H15 -> L3H1 route real, or only inferred?
    So far: 92% of L2H15's effect is nested inside L3H1 (2x2 ablation), it patches back 0%,
    and it mildly perturbs L3H1's key (cos 0.950) and value (cos 0.929) at the acceptor.
    That is an INFERENCE.  The decisive test:

        1. ablate L2H15                                       -> coupling drops
        2. restore L3H1's KEY at the acceptor to its L2H15-INTACT value
        3. restore its VALUE
        4. restore BOTH
        5. how much of the LOST coupling comes back?

    If K+V restores most of it, then
        L2H15 -> {acceptor key, acceptor value} -> L3H1 -> coupling
    is established CAUSALLY, not by nesting + cosine.

    rescue = (coupling_rescued - coupling_L2ablated) / (coupling_baseline - coupling_L2ablated)

QUESTION 2 -- bounded MLP2 mediation (ONE experiment, not a research branch).
    MLP2 sits between L2H15's write and L3H1's read:
        L2H15 -> residual -> MLP2 -> residual -> L3H1's K/V
    Ablate L2H15 and restore MLP2's WRITE at the acceptor to its L2H15-intact value.
        large rescue  -> L2H15 acts THROUGH MLP2   (then, and only then, consider a transcoder)
        small rescue  -> L2H15 writes to the residual directly; use the subspace tools
        mixed         -> report both routes; do not force a narrative

Every patch window is validated against the corruption mask (src/gfmcd/circuits/guards.py).
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
from gfmcd.circuits.guards import (  # noqa: E402
    PatchWindow, acceptor_motif_positions, adjacent_window_controls, validate_all,
)
from gfmcd.circuits.hf_bert import edit_heads  # noqa: E402
from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.coupling import (  # noqa: E402
    ACCEPTOR_ALTS, CouplingConfig, build_coupling_dataset, stable_rng, substitute,
)
from gfmcd.splicing.dataset import PyfaidxGenome  # noqa: E402

L3, H3 = 3, 1
L2, H2 = 2, 15


@contextmanager
def _noop():
    yield


@contextmanager
def patch_head_proj(ad, layer, head, which, clean, pos):
    """Patch head `head`'s query/key/value at token window `pos` [B,K]."""
    mod = getattr(ad.attention_self_modules[layer], which)
    Dh = ad.model.config.hidden_size // ad.num_heads
    lo, hi = head * Dh, (head + 1) * Dh

    def hook(_m, _i, out):
        out = out.clone()
        b = torch.arange(out.shape[0], device=out.device)[:, None].expand_as(pos)
        out[b, pos, lo:hi] = clean[b, pos, lo:hi]
        return out
    h = mod.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


@contextmanager
def patch_mlp_write(ad, layer, clean, pos):
    """Patch the MLP's WRITE into the residual at token window `pos`.

    In HF BERT the MLP's contribution is ``layer.output.dense(intermediate)``, BEFORE the
    residual add + LayerNorm -- so hooking ``.output.dense`` isolates the MLP's write.
    """
    mod = ad.encoder_layers[layer].output.dense

    def hook(_m, _i, out):
        out = out.clone()
        b = torch.arange(out.shape[0], device=out.device)[:, None].expand_as(pos)
        out[b, pos] = clean[b, pos]
        return out
    h = mod.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


def gboot_ratio(num, dnm, genes, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    by = {}
    for a, b, g in zip(num, dnm, genes):
        by.setdefault(g, []).append((a, b))
    k = list(by)
    out = np.empty(n)
    for i in range(n):
        p = rng.choice(len(k), len(k), replace=True)
        pr = [t for j in p for t in by[k[j]]]
        out[i] = sum(t[0] for t in pr) / max(sum(t[1] for t in pr), 1e-9)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default="/Users/cberg/Research/data/genomes/gencode.v50.basic.annotation.gtf.gz")
    ap.add_argument("--fasta", default="/Users/cberg/Research/data/genomes/hg38.fa")
    ap.add_argument("--model", default="data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt")
    ap.add_argument("--out", default="results/splicebert_coupling/development/l2_rescue.json")
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
    X = [x for x in X if x.split == "val" and x.next_acceptor_pos is not None
         and x.sequence[x.donor_pos:x.donor_pos + 2] == "GT"
         and 60 <= (x.next_acceptor_pos - x.donor_pos) <= 150]
    r = stable_rng("rescue", cfg.seed)
    r.shuffle(X)

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

    keep = []
    with torch.no_grad():
        for i in range(0, min(len(X), args.n * 5), args.batch):
            ch = X[i:i + args.batch]
            dn = [x.donor_pos for x in ch]
            e = (metric(ad.model(**enc_of([x.sequence for x in ch], dn)).logits.float(), dn)
                 - metric(ad.model(**enc_of(
                     [substitute(x.sequence, x.next_acceptor_pos - 2, ACCEPTOR_ALTS[0])
                      for x in ch], dn)).logits.float(), dn))
            keep += [x for j, x in enumerate(ch) if float(e[j]) >= 0.3]
            if len(keep) >= args.n:
                break
    X = keep[: args.n]
    genes = [x.gene_id for x in X]
    print(f"{len(X)} held-out exons with a real coupling effect   device={dev}")

    def capture(enc, ctx=_noop):
        """L3's q/k/v projections and layer-2's MLP write."""
        st = {}
        mods = {w: getattr(ad.attention_self_modules[L3], w) for w in ("key", "value")}
        mods["mlp2"] = ad.encoder_layers[L2].output.dense
        hs = [m.register_forward_hook(
            lambda _m, _i, o, w=w: st.__setitem__(w, o.detach())) for w, m in mods.items()]
        try:
            with ctx():
                out = ad.model(**enc)
        finally:
            for h in hs:
                h.remove()
        return st, out

    ABL = lambda: edit_heads(ad, [(L2, H2)], mode="mean")  # noqa: E731
    CONDS = ["K only", "V only", "K+V", "MLP2 write", "K+V @control-pos"]
    num = {c: [] for c in CONDS}
    base_c, abl_c, dnm = [], [], []
    prov = None

    with torch.no_grad():
        for i in range(0, len(X), args.batch):
            ch = X[i:i + args.batch]
            dn = [x.donor_pos for x in ch]
            cl = enc_of([x.sequence for x in ch], dn)
            co = enc_of([substitute(x.sequence, x.next_acceptor_pos - 2, ACCEPTOR_ALTS[0])
                         for x in ch], dn)

            # sources: L2H15 INTACT, on the clean AND the corrupt sequence
            src_cl, o_cl = capture(cl)
            src_co, o_co = capture(co)
            m_cl = metric(o_cl.logits.float(), dn)
            m_co = metric(o_co.logits.float(), dn)
            cbase = m_cl - m_co                                    # baseline coupling

            with ABL():
                m_cl_a = metric(ad.model(**cl).logits.float(), dn)
                m_co_a = metric(ad.model(**co).logits.float(), dn)
            cabl = m_cl_a - m_co_a                                 # coupling with L2H15 gone

            base_c += cbase.tolist()
            abl_c += cabl.tolist()
            dnm += (cbase - cabl).tolist()                         # the coupling L2H15 supplies

            acc = torch.tensor([[x.next_acceptor_pos + OFF + k for k in (-3, -2, -1, 0)]
                                for x in ch], device=ad.device)
            ctl = torch.tensor([[max(4, x.donor_pos + OFF
                                     - (x.next_acceptor_pos - x.donor_pos)) + k
                                 for k in (-3, -2, -1, 0)] for x in ch], device=ad.device)
            if prov is None:
                corr0 = acceptor_motif_positions(ch[0].next_acceptor_pos, OFF)
                w = [PatchWindow("rescue@acceptor", tuple(int(v) for v in acc[0]), corr0),
                     PatchWindow("rescue@control", tuple(int(v) for v in ctl[0]), corr0,
                                 is_control=True)]
                w += adjacent_window_controls(tuple(int(v) for v in acc[0]), corr0)
                prov = validate_all(w)

            specs = {
                "K only":            [("key", acc)],
                "V only":            [("value", acc)],
                "K+V":               [("key", acc), ("value", acc)],
                "K+V @control-pos":  [("key", ctl), ("value", ctl)],
            }
            for c, sp in specs.items():
                with ExitStack() as st:
                    st.enter_context(ABL())
                    for which, pos in sp:
                        st.enter_context(patch_head_proj(ad, L3, H3, which, src_cl[which], pos))
                    mc = metric(ad.model(**cl).logits.float(), dn)
                with ExitStack() as st:
                    st.enter_context(ABL())
                    for which, pos in sp:
                        st.enter_context(patch_head_proj(ad, L3, H3, which, src_co[which], pos))
                    mk = metric(ad.model(**co).logits.float(), dn)
                num[c] += ((mc - mk) - cabl).tolist()

            # MLP2 mediation: restore MLP2's WRITE at the acceptor to its L2H15-intact value
            with ExitStack() as st:
                st.enter_context(ABL())
                st.enter_context(patch_mlp_write(ad, L2, src_cl["mlp2"], acc))
                mc = metric(ad.model(**cl).logits.float(), dn)
            with ExitStack() as st:
                st.enter_context(ABL())
                st.enter_context(patch_mlp_write(ad, L2, src_co["mlp2"], acc))
                mk = metric(ad.model(**co).logits.float(), dn)
            num["MLP2 write"] += ((mc - mk) - cabl).tolist()

    base_c, abl_c, dnm = map(np.asarray, (base_c, abl_c, dnm))
    print("\n" + "=" * 78)
    print("L2H15 RESCUE  (held-out)")
    print(f"  baseline coupling            {base_c.mean():+.3f}")
    print(f"  coupling with L2H15 ablated  {abl_c.mean():+.3f}   (L2H15 supplies "
          f"{dnm.mean():+.3f})")
    print(f"\n  restore L3H1's ... to its L2H15-INTACT value, with L2H15 still ablated:")
    print(f"    {'patch':22s} {'coupling rescued':>17s} {'95% CI':>18s}")
    R = {}
    for c in CONDS:
        v = float(np.sum(num[c]) / np.sum(dnm))
        lo, hi = gboot_ratio(num[c], dnm, genes)
        R[c] = dict(rescue=v, lo=lo, hi=hi)
        print(f"    {c:22s} {v:17.3f}  [{lo:+.3f},{hi:+.3f}]")

    kv = R["K+V"]["rescue"]
    mlp = R["MLP2 write"]["rescue"]
    print(f"\n  => K+V at the acceptor restores {100*kv:.0f}% of the coupling L2H15 supplies")
    if kv > 0.7:
        print("     ROUTE ESTABLISHED CAUSALLY: L2H15 -> {acceptor key, value} -> L3H1 -> coupling")
    elif kv > 0.3:
        print("     PARTIAL: much of L2H15's effect reaches L3H1 via the acceptor's K/V, "
              "but not all")
    else:
        print("     NOT ESTABLISHED: L2H15's effect does NOT reach L3H1 through the "
              "acceptor's K/V")

    print(f"\n  => restoring MLP2's write at the acceptor restores {100*mlp:.0f}%")
    if mlp > 0.7:
        print("     MLP2 MEDIATES the route -> inspect the MLP; a targeted transcoder is warranted")
    elif mlp < 0.3:
        print("     MLP2 does NOT mediate -> L2H15 writes to the residual stream DIRECTLY.")
        print("     Do NOT train a transcoder.  Use the existing subspace tools.")
    else:
        print("     MIXED mediation -- quantify both routes; do not force one narrative")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n": len(X), "baseline_coupling": float(base_c.mean()),
        "coupling_L2H15_ablated": float(abl_c.mean()),
        "coupling_supplied_by_L2H15": float(dnm.mean()),
        "rescue": R, "patch_windows": prov,
    }, indent=1))
    ExperimentManifest.create(
        "splicebert_coupling:l2_rescue", sys.argv,
        config={"n": len(X), "feeder": [L2, H2], "reader": [L3, H3]},
        notes=["Rescue = restore L3H1's K/V (or MLP2's write) at the acceptor to its "
               "L2H15-INTACT value while L2H15 stays ablated.",
               "Bounded MLP2 mediation: no transcoder unless MLP2 is implicated."],
    ).write(out.parent / "l2_rescue.manifest.json")
    print("=" * 78)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
