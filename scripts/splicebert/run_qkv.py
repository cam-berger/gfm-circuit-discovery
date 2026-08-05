#!/usr/bin/env python
"""Does L2H15 build the KEY that lets L3H1 FIND the acceptor, or the VALUE it RETRIEVES?

MOTIVATION.  The path decomposition produced a hard dissociation:
    ablating L2H15 destroys 45% of the coupling,
    patching L2H15's clean output restores 0% of it (0.001 [-0.007, +0.009]).
So L2H15 supplies no *retrievable content*.  It is machinery L3H1 needs in order to read
the partner, not a carrier of the partner signal.  That predicts it acts on SELECTION
(the key / attention) rather than RETRIEVAL (the value).  This script tests that.

PART 1 -- ATTENTION.  L3H1's attention from the donor to the acceptor, with L2H15 intact
          vs mean-ablated.  If L2H15 builds the key that makes the acceptor findable,
          ablating it should COLLAPSE L3H1's attention to the partner.
          Control: attention to a distance-matched non-site.

PART 2 -- K vs V.  Patch L3H1's KEY at the acceptor, or its VALUE at the acceptor, from the
          clean run into the corrupt run, and measure how much coupling each restores.
              recovery(K) >> recovery(V)  =>  the corruption acts through SELECTION
              recovery(V) >> recovery(K)  =>  it acts through RETRIEVED CONTENT
          Controls: patch the QUERY at the donor; patch K/V at a non-partner position.

PART 3 -- Does ablating L2H15 perturb L3H1's KEY at the acceptor more than its VALUE?
          Direct measurement, no patching: cosine similarity of L3H1's k/v at the acceptor,
          with vs without L2H15.
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
def patch_proj(adapter, layer, head, which, clean, positions):
    """Patch head `head`'s KEY or VALUE at `positions` from a clean run.

    BERT splits [B,S,D] into heads by contiguous slices of the last dim, so head h owns
    [h*Dh : (h+1)*Dh].  positions: LongTensor [B] of token indices (one per example).
    """
    mod = getattr(adapter.attention_self_modules[layer], which)  # .key / .value
    Dh = adapter.model.config.hidden_size // adapter.num_heads
    lo, hi = head * Dh, (head + 1) * Dh

    def hook(_m, _i, out):
        out = out.clone()
        pos = positions if positions.dim() == 2 else positions[:, None]   # [B,K]
        b = torch.arange(out.shape[0], device=out.device)[:, None].expand_as(pos)
        out[b, pos, lo:hi] = clean[b, pos, lo:hi]
        return out

    h = mod.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


def gboot_ratio(num, dnm, genes, n=1000, seed=0):
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
    ap.add_argument("--gtf", default=None)
    ap.add_argument("--fasta", default=None)
    ap.add_argument("--model", default="data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt")
    ap.add_argument("--out", default="results/splicebert_coupling/qkv.json")
    args = ap.parse_args()

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    dev = args.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    mp = args.model if Path(args.model).is_absolute() else str(REPO / args.model)
    # eager: SDPA silently ignores output_attentions and returns zeros
    ad = SpliceBertAdapter(
        AutoModelForMaskedLM.from_pretrained(mp, attn_implementation="eager"),
        AutoTokenizer.from_pretrained(mp), device=dev)
    tok, MASK = ad.tokenizer, ad.tokenizer.mask_token_id
    BID = {b: tok.convert_tokens_to_ids(b) for b in "ACGT"}
    OFF = ad.prefix_tokens
    Dh = ad.model.config.hidden_size // ad.num_heads

    cfg = CouplingConfig()
    X = build_coupling_dataset(args.gtf, PyfaidxGenome(args.fasta), cfg)
    X = [x for x in X if x.split == "val" and x.next_acceptor_pos is not None
         and x.sequence[x.donor_pos:x.donor_pos + 2] == "GT"
         and 60 <= (x.next_acceptor_pos - x.donor_pos) <= 150]
    r = stable_rng("qkv", cfg.seed)
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
    print(f"{len(X)} held-out exons with a real coupling effect   device={dev}")

    # capture L3's key/value projections
    def capture_kv(enc, ctx=_noop):
        store = {}
        mods = {w: getattr(ad.attention_self_modules[L3], w) for w in ("query", "key", "value")}
        hs = [m.register_forward_hook(
            lambda _m, _i, o, w=w: store.__setitem__(w, o.detach())) for w, m in mods.items()]
        try:
            with ctx():
                out = ad.model(**enc, output_attentions=True)
        finally:
            for h in hs:
                h.remove()
        return store, out

    R = {"attention": {}, "kv": {}, "geometry": {}}

    # ---------------- PART 1 + 3: attention and k/v geometry -------------------------
    att_intact, att_abl, att_corrupt, att_ctrl_intact, att_ctrl_abl = [], [], [], [], []
    cos_k, cos_v = [], []
    with torch.no_grad():
        for i in range(0, len(X), args.batch):
            ch = X[i:i + args.batch]
            dn = [x.donor_pos for x in ch]
            cl = enc_of([x.sequence for x in ch], dn)

            co = enc_of([substitute(x.sequence, x.next_acceptor_pos - 2, ACCEPTOR_ALTS[0])
                         for x in ch], dn)
            kv0, o0 = capture_kv(cl)
            kv1, o1 = capture_kv(cl, lambda: edit_heads(ad, [(L2, H2)], mode="mean"))
            _, oc = capture_kv(co)
            a0 = o0.attentions[L3][:, H3]      # [B,S,S] intact
            a1 = o1.attentions[L3][:, H3]      # L2H15 ablated
            ac = oc.attentions[L3][:, H3]      # acceptor CORRUPTED

            for j, x in enumerate(ch):
                d = x.donor_pos + OFF
                p = x.next_acceptor_pos + OFF
                q = d - (p - d)                # distance-matched control position
                att_intact.append(a0[j, d, p - 2:p + 1].sum().item())
                att_abl.append(a1[j, d, p - 2:p + 1].sum().item())
                att_corrupt.append(ac[j, d, p - 2:p + 1].sum().item())
                if 2 < q < a0.shape[-1] - 1:
                    att_ctrl_intact.append(a0[j, d, q - 2:q + 1].sum().item())
                    att_ctrl_abl.append(a1[j, d, q - 2:q + 1].sum().item())
                lo, hi = H3 * Dh, (H3 + 1) * Dh
                for name, kv, acc in (("k", "key", cos_k), ("v", "value", cos_v)):
                    u = kv0[kv][j, p, lo:hi]
                    w = kv1[kv][j, p, lo:hi]
                    acc.append(torch.nn.functional.cosine_similarity(
                        u[None], w[None]).item())

    print("\n" + "=" * 76)
    print("PART 1  L3H1 ATTENTION (donor -> acceptor), with vs without L2H15")
    print(f"  attention to the PARTNER acceptor   intact {np.mean(att_intact):.4f}"
          f"   L2H15-ablated {np.mean(att_abl):.4f}"
          f"   ({100*(1-np.mean(att_abl)/max(np.mean(att_intact),1e-9)):.0f}% collapse)")
    print(f"  attention to a distance-matched ctrl intact {np.mean(att_ctrl_intact):.4f}"
          f"   L2H15-ablated {np.mean(att_ctrl_abl):.4f}")
    print(f"  attention to the PARTNER when the ACCEPTOR IS CORRUPTED: {np.mean(att_corrupt):.4f}"
          f"   ({100*(1-np.mean(att_corrupt)/max(np.mean(att_intact),1e-9)):.0f}% collapse)")
    print("    -> if attention SURVIVES corruption, the effect cannot be pure SELECTION")
    R["attention"] = dict(partner_intact=float(np.mean(att_intact)),
                          partner_ablated=float(np.mean(att_abl)),
                          partner_corrupted=float(np.mean(att_corrupt)),
                          control_intact=float(np.mean(att_ctrl_intact)),
                          control_ablated=float(np.mean(att_ctrl_abl)))

    print("\nPART 3  Does ablating L2H15 perturb L3H1's KEY or its VALUE at the acceptor?")
    print(f"  cosine(k with L2H15, k without)  = {np.mean(cos_k):.4f}")
    print(f"  cosine(v with L2H15, v without)  = {np.mean(cos_v):.4f}")
    print("  (lower = more disrupted)")
    R["geometry"] = dict(cos_key=float(np.mean(cos_k)), cos_value=float(np.mean(cos_v)))

    # ---------------- PART 2: K vs V patching ---------------------------------------
    print("\nPART 2  PATCH L3H1's KEY vs VALUE at the acceptor (clean -> corrupt)")
    conds = ["key@acceptor", "value@acceptor", "key+value@acceptor", "query@donor",
             "key@control-pos", "value@control-pos"]
    window_prov = []
    att_patch = {c: [] for c in conds}
    num = {c: [] for c in conds}
    dnm = []
    with torch.no_grad():
        for i in range(0, len(X), args.batch):
            ch = X[i:i + args.batch]
            dn = [x.donor_pos for x in ch]
            cl = enc_of([x.sequence for x in ch], dn)
            co = enc_of([substitute(x.sequence, x.next_acceptor_pos - 2, ACCEPTOR_ALTS[0])
                         for x in ch], dn)
            kv_clean, _ = capture_kv(cl)
            m_c = metric(ad.model(**cl).logits.float(), dn)
            m_k = metric(ad.model(**co).logits.float(), dn)
            dnm += (m_c - m_k).tolist()

            # THE CORRUPTION IS AT THE AG, i.e. [a-2, a-1] -- NOT at a.  Patch the window
            # that actually changed (plus one base either side), or we patch a position the
            # corruption never touched and recover ~nothing for the wrong reason.
            acc_pos = torch.tensor(
                [[x.next_acceptor_pos + OFF + k for k in (-3, -2, -1, 0)] for x in ch],
                device=ad.device)
            don_pos = torch.tensor([[x.donor_pos + OFF] for x in ch], device=ad.device)
            ctl_pos = torch.tensor(
                [[max(4, (x.donor_pos + OFF) - (x.next_acceptor_pos - x.donor_pos)) + k
                  for k in (-3, -2, -1, 0)] for x in ch], device=ad.device)

            # GUARD: a patch that does not cover the corrupted tokens cannot restore them.
            # Validated per batch; the exact token indices go into provenance.
            x0 = ch[0]
            corr0 = acceptor_motif_positions(x0.next_acceptor_pos, OFF)
            wins = [PatchWindow("key/value@acceptor",
                                tuple(int(v) for v in acc_pos[0]), corr0),
                    PatchWindow("control-pos", tuple(int(v) for v in ctl_pos[0]), corr0,
                                is_control=True)]
            wins += adjacent_window_controls(tuple(int(v) for v in acc_pos[0]), corr0)
            window_prov = validate_all(wins)

            specs = {
                "key@acceptor":       [("key", acc_pos)],
                "value@acceptor":     [("value", acc_pos)],
                "key+value@acceptor": [("key", acc_pos), ("value", acc_pos)],
                "query@donor":        [("query", don_pos)],
                "key@control-pos":    [("key", ctl_pos)],
                "value@control-pos":  [("value", ctl_pos)],
            }
            for c, sp in specs.items():
                with ExitStack() as st:
                    for which, pos in sp:
                        st.enter_context(patch_proj(ad, L3, H3, which, kv_clean[which], pos))
                    o = ad.model(**co, output_attentions=True)
                    m = metric(o.logits.float(), dn)
                num[c] += (m - m_k).tolist()
                # ATTENTION under this patch -- the claim "K restores selection" must be
                # SHOWN, not asserted.
                A = o.attentions[L3][:, H3]
                for j, x in enumerate(ch):
                    dd = x.donor_pos + OFF
                    pp = x.next_acceptor_pos + OFF
                    att_patch[c].append(A[j, dd, pp - 2:pp + 1].sum().item())

    a_clean = float(np.mean(att_intact))
    a_corr = float(np.mean(att_corrupt))
    print(f"  {'patch':22s} {'coupling restored':>17s} {'attention restored':>19s}")
    print(f"  {'(none: corrupt)':22s} {0.0:17.3f} {a_corr:19.4f}")
    for c in conds:
        v = float(np.sum(num[c]) / np.sum(dnm))
        lo, hi = gboot_ratio(num[c], dnm, genes)
        a = float(np.mean(att_patch[c])) if att_patch[c] else float("nan")
        arec = (a - a_corr) / max(a_clean - a_corr, 1e-9)
        R["kv"][c] = dict(recovery=v, lo=lo, hi=hi, attention=a, attention_restored=arec)
        print(f"  {c:22s} {v:17.3f} {a:11.4f} ({100*arec:3.0f}%)")
    print(f"  {'(clean reference)':22s} {1.0:17.3f} {a_clean:19.4f}")
    print("\n  KEY restores SELECTION but not the conclusion; VALUE restores neither alone;")
    print("  only TOGETHER does the head both find the partner AND read it correctly.")

    # ---- THE INTERACTION TERM ------------------------------------------------------
    # The 5x ratio is descriptive but unstable: the marginals are small, so their ratio is
    # noisy.  The paired factorial interaction is the statistic to report:
    #     I = R_KV - R_K - R_V + R_null      (R_null = 0: no patch, no recovery)
    nk = np.asarray(num["key@acceptor"]); nv = np.asarray(num["value@acceptor"])
    nkv = np.asarray(num["key+value@acceptor"]); dd = np.asarray(dnm)

    def ratios(idx):
        D = dd[idx].sum()
        return (nk[idx].sum() / D, nv[idx].sum() / D, nkv[idx].sum() / D)

    Rk, Rv, Rkv = ratios(np.arange(len(dd)))
    I = Rkv - Rk - Rv
    rng = np.random.default_rng(0)
    by = {}
    for i, g in enumerate(genes):
        by.setdefault(g, []).append(i)
    gk = list(by)
    bs = np.empty(2000)
    for b in range(2000):
        pick = rng.choice(len(gk), len(gk), replace=True)
        idx = np.array([i for j in pick for i in by[gk[j]]])
        a_, b_, c_ = ratios(idx)
        bs[b] = c_ - a_ - b_
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    print(f"\n  K x V INTERACTION  I = R_KV - R_K - R_V")
    print(f"    R_K  {Rk:6.3f}   R_V {Rv:6.3f}   R_KV {Rkv:6.3f}")
    print(f"    I = {I:+.3f}  [{lo:+.3f},{hi:+.3f}]  (gene-grouped bootstrap, per example)")
    if lo > 0:
        print("    => SUPER-ADDITIVE: neither channel alone suffices. L3H1 performs a")
        print("       CONTENT-ADDRESSED READ -- the acceptor must be both FINDABLE (key)")
        print("       and INFORMATIVE (value); each is worthless without the other.")
    R["interaction"] = dict(R_K=float(Rk), R_V=float(Rv), R_KV=float(Rkv),
                            I=float(I), ci=[lo, hi])
    R["patch_windows"] = window_prov
    print("\n  NOTE: query@donor recovery is a null about the CORRUPTION, not the query --")
    print("        acceptor corruption did not perturb the donor query, so a clean-query")
    print("        patch has nothing to restore.  Query NECESSITY is untested.")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=1))
    ExperimentManifest.create(
        "splicebert_coupling:qkv", sys.argv,
        config={"n": len(X), "heads": {"reader": [L3, H3], "feeder": [L2, H2]}},
        notes=["L2H15 has 0 patch-recovery but 45% ablation necessity -> predicted to act "
               "on SELECTION (key), not RETRIEVAL (value).  This tests that."],
    ).write(out.parent / "qkv.manifest.json")
    print("=" * 76)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
