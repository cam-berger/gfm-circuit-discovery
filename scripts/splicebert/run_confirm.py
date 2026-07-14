#!/usr/bin/env python
"""PUBLISHED-LEGACY 2026-07-12 -- superseded by run_sealed_test.py.  Two numbers are RETRACTED.

Do not re-cite this script's output:
  * mlm_health (0.9958 -> 0.9953) is VACUOUS (R8).  Its probe positions are never masked, so it
    scores the model COPYING ITS OWN INPUT.  It returns ~0.99 for everything and RISES to 0.999
    with all 96 heads ablated.  Use CouplingTask.mlm_health (masked probes): baseline ~0.53.
  * p_add_one = 0.000999 is a FUNCTION OF --n-null, not of the data (R9).  1000 draws with
    replacement from a 95-head universe at k=1 are 95 distinct values; the honest p is 1/96 =
    0.0104.  matched_random_head_sets now de-duplicates.
  * val_coupling = 1.359 is a CONDITIONAL mean (filtered to coupling >= 0.3), reported as a
    behavioural magnitude (R10).  The unconditional value is 0.973 [0.919, 1.023].
  * Everything here ran on PLUS-STRAND DATA ONLY (R11).

The ratios it reports (recovery 0.808, rank 1/96, subspace fractions) are unaffected in kind,
but were computed on plus-strand data and must be labelled conditional.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.adapters.splicebert import SpliceBertAdapter  # noqa: E402
from gfmcd.circuits.hf_bert import (  # noqa: E402
    _context_from_output, _merge_context, _replace_output, _split_context,
    capture_head_outputs, edit_heads, matched_random_head_sets, patch_heads,
)
from gfmcd.rigor import empirical_upper_p_value  # noqa: E402
from gfmcd.splicing.coupling import (  # noqa: E402
    ACCEPTOR_ALTS, CouplingConfig, build_coupling_dataset, stable_rng, substitute,
)
from gfmcd.splicing.dataset import PyfaidxGenome  # noqa: E402

LAYER, HEAD = 3, 1


@contextmanager
def _noop():
    yield


@contextmanager
def project_out(adapter, layer, head, U):
    def hook(_m, _i, output):
        ctx = _context_from_output(output)
        sp = _split_context(ctx, adapter.num_heads).clone()
        z = sp[:, head]
        sp[:, head] = z - (z @ U) @ U.transpose(0, 1)
        return _replace_output(output, _merge_context(sp))
    h = adapter.attention_self_modules[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


def gboot(vals, genes, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    by = {}
    for v, g in zip(vals, genes):
        if not np.isnan(v):
            by.setdefault(g, []).append(v)
    k = list(by)
    if not k:
        return (np.nan, np.nan)
    m = np.empty(n)
    for i in range(n):
        p = rng.choice(len(k), len(k), replace=True)
        m[i] = np.mean([v for j in p for v in by[k[j]]])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


class Task:
    def __init__(self, adapter, bs=8):
        self.a, self.bs = adapter, bs
        self.mask = adapter.tokenizer.mask_token_id
        self.bid = {b: adapter.tokenizer.convert_tokens_to_ids(b) for b in "ACGT"}
        self.off = adapter.prefix_tokens

    def ids(self, seqs, dn):
        e = self.a.tokenizer([" ".join(s) for s in seqs], return_tensors="pt",
                             padding=True).to(self.a.device)
        for j, d in enumerate(dn):
            e["input_ids"][j, d + self.off] = self.mask
        return e

    def pair(self, ch):
        dn = [x.donor_pos for x in ch]
        return (self.ids([x.sequence for x in ch], dn),
                self.ids([substitute(x.sequence, x.next_acceptor_pos - 2, ACCEPTOR_ALTS[0])
                          for x in ch], dn), dn)

    def metric(self, lg, dn):
        i = torch.arange(len(dn), device=lg.device)
        p = torch.tensor([d + self.off for d in dn], device=lg.device)
        v = lg[i, p]
        return v[:, self.bid["G"]] - torch.stack(
            [v[:, self.bid[b]] for b in "ACT"], -1).max(-1).values


def load(args, split, adapter, task, n):
    cfg = CouplingConfig()
    X = build_coupling_dataset(args.gtf, PyfaidxGenome(args.fasta), cfg)
    X = [x for x in X if x.split == split and x.next_acceptor_pos is not None
         and x.sequence[x.donor_pos:x.donor_pos + 2] == "GT"
         and 60 <= (x.next_acceptor_pos - x.donor_pos) <= 150]
    r = stable_rng(f"conf-{split}", cfg.seed)
    r.shuffle(X)
    keep, eff = [], []
    with torch.no_grad():
        for i in range(0, min(len(X), n * 5), task.bs):
            ch = X[i:i + task.bs]
            cl, co, dn = task.pair(ch)
            e = (task.metric(adapter.model(**cl).logits.float(), dn)
                 - task.metric(adapter.model(**co).logits.float(), dn))
            for j, x in enumerate(ch):
                if float(e[j]) >= 0.3:
                    keep.append(x)
                    eff.append(float(e[j]))
            if len(keep) >= n:
                break
    return keep[:n], eff[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--n-scan", type=int, default=120)
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--n-fit", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default="/Users/cberg/Research/data/genomes/gencode.v50.basic.annotation.gtf.gz")
    ap.add_argument("--fasta", default="/Users/cberg/Research/data/genomes/hg38.fa")
    ap.add_argument("--model", default="data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt")
    ap.add_argument("--out", default="results/splicebert_coupling/confirmation.json")
    args = ap.parse_args()

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    dev = args.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    mp = args.model if Path(args.model).is_absolute() else str(REPO / args.model)
    adapter = SpliceBertAdapter(AutoModelForMaskedLM.from_pretrained(mp),
                                AutoTokenizer.from_pretrained(mp), device=dev)
    task = Task(adapter, bs=args.batch)
    L, H = adapter.num_layers, adapter.num_heads
    DH = adapter.model.config.hidden_size // H
    R = {}

    VAL, veff = load(args, "val", adapter, task, args.n_val)
    genes = [x.gene_id for x in VAL]
    lo, hi = gboot(veff, genes)
    print(f"HELD-OUT val (chr11/16/19): {len(VAL)} exons   "
          f"coupling {np.mean(veff):+.3f} [{lo:+.3f},{hi:+.3f}]  (test chr8/9 SEALED)")
    R["val_coupling"] = dict(n=len(VAL), mean=float(np.mean(veff)), ci=[lo, hi])

    def run(exs, ctx, per_ex=False):
        C, K = [], []
        with torch.no_grad():
            for i in range(0, len(exs), task.bs):
                cl, co, dn = task.pair(exs[i:i + task.bs])
                with ctx():
                    C += task.metric(adapter.model(**cl).logits.float(), dn).tolist()
                    K += task.metric(adapter.model(**co).logits.float(), dn).tolist()
        return (np.array(C), np.array(K)) if per_ex else (float(np.mean(C)), float(np.mean(K)))

    # ---- B. full 96-head scan on HELD-OUT val --------------------------------------
    SC = VAL[: args.n_scan]
    heads = [(l, h) for l in range(L) for h in range(H)]
    den = np.zeros((len(heads), len(SC)))
    e0 = 0
    for i in range(0, len(SC), args.batch):
        ch = SC[i:i + args.batch]
        cl, co, dn = task.pair(ch)
        with torch.no_grad():
            cc = capture_head_outputs(adapter, lambda: adapter.model(**cl))
            mc = task.metric(adapter.model(**cl).logits.float(), dn)
            mk = task.metric(adapter.model(**co).logits.float(), dn)
            dnm = (mc - mk).clamp_min(1e-3)
            for hi_, hd in enumerate(heads):
                with patch_heads(adapter, cc, [hd]):
                    m = task.metric(adapter.model(**co).logits.float(), dn)
                den[hi_, e0:e0 + len(ch)] = ((m - mk) / dnm).cpu().numpy()
        e0 += len(ch)
        print(f"  scan {e0}/{len(SC)}", end="\r", flush=True)
    print()
    dm = den.mean(1)
    order = np.argsort(-dm)
    print("\nB. FULL 96-HEAD DENOISE SCAN on HELD-OUT val  (top 6)")
    for i in order[:6]:
        l, h = heads[i]
        lo_, hi_ = gboot(den[i], [x.gene_id for x in SC])
        star = "  <-- discovered on train" if (l, h) == (LAYER, HEAD) else ""
        print(f"    L{l}H{h:<3d} recovery {dm[i]:6.3f}  [{lo_:+.3f},{hi_:+.3f}]{star}")
    rank = int(np.where(order == heads.index((LAYER, HEAD)))[0][0]) + 1
    print(f"    => L{LAYER}H{HEAD} ranks #{rank} of 96 on held-out data")
    R["scan_rank_of_L3H1"] = rank
    R["scan_top"] = [dict(layer=heads[i][0], head=heads[i][1], recovery=float(dm[i]))
                     for i in order[:8]]

    # ---- C. 2x2 factorial on val, with gene-grouped bootstrap -----------------------
    ci, xi = run(VAL, _noop, per_ex=True)
    ca, xa = run(VAL, lambda: edit_heads(adapter, [(LAYER, HEAD)], mode="mean"), per_ex=True)
    did_v = (ci - xi) - (ca - xa)
    loc_v = xi - xa
    dlo, dhi = gboot(did_v, genes)
    llo, lhi = gboot(loc_v, genes)
    print("\nC. 2x2 FACTORIAL on HELD-OUT val")
    print(f"    {'':22s} {'L3H1 intact':>13s} {'L3H1 ablated':>14s}")
    print(f"    {'partner intact':22s} {ci.mean():13.3f} {ca.mean():14.3f}")
    print(f"    {'partner corrupted':22s} {xi.mean():13.3f} {xa.mean():14.3f}")
    print(f"    partner-dependent (DiD) : {did_v.mean():+.3f}  [{dlo:+.3f},{dhi:+.3f}]")
    print(f"    local  (partner-indep)  : {loc_v.mean():+.3f}  [{llo:+.3f},{lhi:+.3f}]"
          f"   <- CI containing 0 => no local role")
    R["factorial"] = dict(M_CI=float(ci.mean()), M_XI=float(xi.mean()),
                          M_CA=float(ca.mean()), M_XA=float(xa.mean()),
                          did=float(did_v.mean()), did_ci=[dlo, dhi],
                          local=float(loc_v.mean()), local_ci=[llo, lhi])

    # ---- D. matched-random head-set null, 1000 sets ---------------------------------
    NL = VAL[:100]
    b_c, b_k = run(NL, _noop)
    a_c, a_k = run(NL, lambda: edit_heads(adapter, [(LAYER, HEAD)], mode="mean"))
    obs = (b_c - b_k) - (a_c - a_k)
    null_sets = matched_random_head_sets(adapter, [(LAYER, HEAD)], n=args.n_null, seed=0,
                                         exclude_selected=True)
    drops = []
    for i, s in enumerate(null_sets):
        c, k = run(NL, lambda s=s: edit_heads(adapter, list(s), mode="mean"))
        drops.append((b_c - b_k) - (c - k))
        if i % 100 == 0:
            print(f"  null {i}/{args.n_null}", end="\r", flush=True)
    print()
    p = empirical_upper_p_value(obs, drops)
    print(f"\nD. MATCHED-RANDOM NULL ({args.n_null} single-head sets, n={len(NL)} exons)")
    print(f"    L3H1 coupling drop {obs:+.3f}  vs random mean {np.mean(drops):+.3f} "
          f"(max {np.max(drops):+.3f})   add-one p = {p:.5f}")
    R["null"] = dict(observed=float(obs), mean=float(np.mean(drops)),
                     max=float(np.max(drops)), p_add_one=float(p), n=len(drops))

    # ---- E. subspace: FIT ON TRAIN, SCORE ON VAL ------------------------------------
    TR, _ = load(args, "train", adapter, task, args.n_fit)
    deltas = []
    with torch.no_grad():
        for i in range(0, len(TR), args.batch):
            ch = TR[i:i + args.batch]
            cl, co, dn = task.pair(ch)
            cc = capture_head_outputs(adapter, lambda: adapter.model(**cl))
            kk = capture_head_outputs(adapter, lambda: adapter.model(**co))
            for j, x in enumerate(ch):
                pp = x.donor_pos + task.off
                deltas.append((cc[LAYER][j, HEAD, pp] - kk[LAYER][j, HEAD, pp]).cpu().numpy())
    D = np.stack(deltas)
    _, S, V = np.linalg.svd(D - D.mean(0, keepdims=True), full_matrices=False)
    ev = (S ** 2) / (S ** 2).sum()
    print(f"\nE. SUBSPACE (fit on {len(TR)} TRAIN exons, scored on HELD-OUT val)")
    print(f"    delta variance in top dims: {' '.join(f'{e:.2f}' for e in ev[:5])}")
    base = float((ci - xi).mean())
    sub = []
    grand = stable_rng("rand", 0)
    for k in (1, 2, 4):
        U = torch.tensor(V[:k].T, dtype=torch.float32, device=adapter.device)
        c, kk_ = run(VAL, lambda: project_out(adapter, LAYER, HEAD, U))
        rc = []
        for _ in range(10):
            g = torch.Generator().manual_seed(grand.randrange(1 << 30))
            Q, _ = torch.linalg.qr(torch.randn(DH, k, generator=g))
            c2, k2 = run(VAL, lambda: project_out(adapter, LAYER, HEAD, Q.to(adapter.device)))
            rc.append(c2 - k2)
        print(f"    rank {k}: coupling {c-kk_:+.3f} (clean margin {c:.3f})   "
              f"random rank-{k}: coupling {np.mean(rc):+.3f}")
        sub.append(dict(rank=k, coupling=float(c - kk_), margin=float(c),
                        random_coupling=float(np.mean(rc))))
    R["subspace"] = dict(explained_variance=ev[:6].tolist(), baseline_coupling=base, ranks=sub)

    # ---- F. MLM health for L3H1 ALONE ----------------------------------------------
    def health(ctx):
        acc = []
        r = stable_rng("h", 0)
        with torch.no_grad():
            for i in range(0, len(NL), task.bs):
                ch = NL[i:i + task.bs]
                cl, _, _ = task.pair(ch)
                with ctx():
                    lg = adapter.model(**cl).logits.float()
                for j, x in enumerate(ch):
                    ps = [r.randrange(10, len(x.sequence) - 10) for _ in range(40)]
                    pr = lg[j, [p + task.off for p in ps]].argmax(-1)
                    tg = torch.tensor([task.bid[x.sequence[p]] for p in ps], device=pr.device)
                    acc.append((pr == tg).float().mean().item())
        return float(np.mean(acc))
    h0 = health(_noop)
    h1 = health(lambda: edit_heads(adapter, [(LAYER, HEAD)], mode="mean"))
    print(f"\nF. MLM HEALTH (L3H1 alone):  baseline top-1 {h0:.4f} -> ablated {h1:.4f}")
    R["mlm_health"] = dict(baseline=h0, ablated=h1)

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=1))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
