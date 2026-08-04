#!/usr/bin/env python
"""EXHAUSTIVE 96-head ABLATION scan on held-out data, head IDs retained.

WHY THIS EXISTS
---------------
The 1000-set matched-random "null" was NOT a null.  Its tail (max drop +0.794 vs L3H1's
+1.193) is not noise -- if other heads genuinely belong to the circuit, that tail is
MECHANISM, and testing L3H1 against a population that contains the rest of the circuit is
circular.  So we replace it with:

  1. the EXHAUSTIVE causal distribution over all 96 heads, with IDs kept, and
  2. a per-head SPECIFICITY null: the same ablation, but scored on a perturbation of a
     DISTANCE-MATCHED NON-SITE instead of the true partner.  A head that belongs to the
     coupling circuit must lose the PARTNER effect specifically, not degrade everything.

Do NOT identify the tail head from the denoise ranking.  Denoising (patch clean -> corrupt)
measures SUFFICIENCY under restoration; ablation measures NECESSITY under removal.  A
downstream mediator can score low on one and high on the other -- which is precisely the
hypothesis in question, so the two must be measured separately.

Reports, per head:
    drop_partner  = coupling(baseline) - coupling(head ablated)          [necessity]
    drop_control  = pseudo-coupling(baseline) - pseudo-coupling(ablated) [specificity null]
where pseudo-coupling perturbs a distance-matched NON-site instead of the partner.
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
from gfmcd.circuits.hf_bert import edit_heads  # noqa: E402
from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.coupling import (  # noqa: E402
    ACCEPTOR_ALTS, CouplingConfig, build_coupling_dataset, perturb_random, stable_rng,
    substitute,
)
from gfmcd.splicing.dataset import PyfaidxGenome  # noqa: E402


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
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=None)
    ap.add_argument("--fasta", default=None)
    ap.add_argument("--model", default="data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt")
    ap.add_argument("--out", default="results/splicebert_coupling/development/ablation_scan.json")
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
    L, H = ad.num_layers, ad.num_heads

    cfg = CouplingConfig()
    X = build_coupling_dataset(args.gtf, PyfaidxGenome(args.fasta), cfg)
    X = [x for x in X if x.split == args.split and x.next_acceptor_pos is not None
         and x.sequence[x.donor_pos:x.donor_pos + 2] == "GT"
         and 60 <= (x.next_acceptor_pos - x.donor_pos) <= 150]
    r = stable_rng("abl", cfg.seed)
    r.shuffle(X)

    # three sequence variants per exon, all measured at the SAME masked donor:
    #   clean | partner destroyed | distance-matched NON-site destroyed (the specificity null)
    rows = []
    for x in X:
        s, d, na = x.sequence, x.donor_pos, x.next_acceptor_pos
        off = na - d
        rr = stable_rng(x.example_id, 1)
        cand = [q for q in (d - off, d + off // 2, d - off // 2)
                if 12 < q < len(s) - 12 and s[q - 2:q] != "AG" and s[q:q + 2] not in ("GT", "GC")]
        if not cand:
            continue
        q = cand[rr.randrange(len(cand))]
        rows.append((s, substitute(s, na - 2, ACCEPTOR_ALTS[0]), perturb_random(s, q),
                     d, x.gene_id))
        if len(rows) >= args.n:
            break
    print(f"{len(rows)} held-out exons ({args.split})   device={dev}   {L*H} heads")

    def margins(seqs, dn, ctx):
        out = []
        with torch.no_grad():
            for i in range(0, len(seqs), args.batch):
                S, P = seqs[i:i + args.batch], dn[i:i + args.batch]
                enc = tok([" ".join(z) for z in S], return_tensors="pt",
                          padding=True).to(ad.device)
                for j, p in enumerate(P):
                    enc["input_ids"][j, p + OFF] = MASK
                with ctx():
                    lg = ad.model(**enc).logits.float()
                for j, p in enumerate(P):
                    v = lg[j, p + OFF]
                    o = max(v[BID[b]].item() for b in "ACT")
                    out.append(v[BID["G"]].item() - o)
        return np.asarray(out)

    clean, pert, ctrl, dn, genes = (list(z) for z in zip(*rows))

    def couplings(ctx):
        c = margins(clean, dn, ctx)
        p = margins(pert, dn, ctx)
        k = margins(ctrl, dn, ctx)
        return c - p, c - k          # (partner coupling, pseudo-coupling) per example

    base_p, base_c = couplings(_noop)
    lo, hi = gboot(base_p, genes)
    print(f"baseline: partner coupling {base_p.mean():+.3f} [{lo:+.3f},{hi:+.3f}]   "
          f"non-site pseudo-coupling {base_c.mean():+.3f}")

    heads = [(l, h) for l in range(L) for h in range(H)]
    res = []
    for i, hd in enumerate(heads):
        p, c = couplings(lambda hd=hd: edit_heads(ad, [hd], mode="mean"))
        dp = base_p - p          # per-example drop in the PARTNER coupling
        dc = base_c - c          # per-example drop in the NON-SITE pseudo-coupling
        res.append(dict(layer=hd[0], head=hd[1],
                        drop_partner=float(dp.mean()), drop_control=float(dc.mean()),
                        specificity=float(dp.mean() - dc.mean()),
                        dp=dp.tolist()))
        print(f"  {i+1}/{len(heads)}  L{hd[0]}H{hd[1]}", end="\r", flush=True)
    print()

    res.sort(key=lambda z: -z["drop_partner"])
    print("\n" + "=" * 78)
    print("EXHAUSTIVE 96-HEAD ABLATION (held-out).  drop_partner = necessity for the coupling")
    print(f"  {'head':7s} {'drop_partner':>13s} {'95% CI':>18s} {'drop_control':>13s} {'specific':>9s}")
    for z in res[:12]:
        lo, hi = gboot(np.array(z["dp"]), genes)
        print(f"  L{z['layer']}H{z['head']:<4d} {z['drop_partner']:13.3f} "
              f"[{lo:+.3f},{hi:+.3f}] {z['drop_control']:13.3f} {z['specificity']:9.3f}")
    tot = sum(max(z["drop_partner"], 0) for z in res)
    print(f"\n  top head = {100*res[0]['drop_partner']/max(tot,1e-9):.0f}% of the total "
          f"positive causal mass;  top-3 = "
          f"{100*sum(max(z['drop_partner'],0) for z in res[:3])/max(tot,1e-9):.0f}%")
    lay = {}
    for z in res:
        lay.setdefault(z["layer"], 0.0)
        lay[z["layer"]] += max(z["drop_partner"], 0)
    print("  causal mass by layer: " + "  ".join(
        f"L{l}={100*v/max(tot,1e-9):.0f}%" for l, v in sorted(lay.items())))
    print("=" * 78)

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "split": args.split, "n": len(rows),
        "baseline_partner_coupling": float(base_p.mean()),
        "baseline_pseudo_coupling": float(base_c.mean()),
        "heads": [{k: v for k, v in z.items() if k != "dp"} for z in res],
    }, indent=1))
    ExperimentManifest.create(
        "splicebert_coupling:ablation_scan", sys.argv,
        config={"n": len(rows), "split": args.split},
        notes=["Exhaustive per-head ABLATION (necessity), IDs retained.",
               "Replaces the matched-random head null, which was not a null: if other "
               "heads belong to the circuit, its tail is mechanism, not noise.",
               "Specificity null = same ablation scored on a distance-matched NON-site."],
    ).write(out.parent / "ablation_scan.manifest.json")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
