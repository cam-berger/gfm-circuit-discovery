#!/usr/bin/env python
"""AUDIT: is gate 5 (b_LATTICE > b_MATCHED-RANDOM) passed by construction?

Two checks:
 A. the actual local attention profile f(m) of the lattice heads.  The attack asserts
    "rolling a lattice head by delta = 3 maps its read set ONTO ITSELF (same absolute
    columns, reweighted)".  That is only true if f is spread over MANY offsets in its
    residue class.  If f is CONCENTRATED (e.g. all mass at m = -3), a roll by 3 relocates
    the dominant source entirely and the 'onto itself' claim is vacuous.
 B. the real thing: damage(delta) for LATTICE and for MATCHED-RANDOM draws, fit
        damage = a*delta + b*1[delta % 3 != 0] + c
    and see whether b_MR is pinned at 0 the way the attack claims.
"""
from __future__ import annotations
import json, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

REPO = Path("/Users/cberg/Research/projects/gfm-circuit-discovery")
sys.path.insert(0, str(REPO / "src"))
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, load_adapter, load_examples,
    noop, pick_device, roll_attention,
)

MASK_STRIDE = 7
MIN_LEN = 60
N = 12
SPAN = 24

dev = pick_device(None)
ad = load_adapter(DEFAULT_MODEL, dev, eager=True)
task = CouplingTask(ad, batch=4)
OFF = task.off
inv = json.loads((REPO / "results/phase-lattice/phase_inventory.json").read_text())
LAT = [(r["layer"], r["head"]) for r in inv["heads"] if r["lattice"]]
NON = [(r["layer"], r["head"]) for r in inv["heads"] if not r["lattice"]]
LAT_BY_L = defaultdict(int)
for l, h in LAT:
    LAT_BY_L[l] += 1
print(f"device={dev}  lattice={len(LAT)}  non-lattice={len(NON)}")
print("lattice heads per layer:", dict(sorted(LAT_BY_L.items())))

X, _ = load_examples(DEFAULT_GTF, DEFAULT_FASTA, "val", N, task=task, tag="audit-g5",
                     min_effect=None)
X = [x for x in X if x.exon_length >= MIN_LEN and x.next_acceptor_pos - x.donor_pos >= MIN_LEN]
print(f"{len(X)} examples\n")

# ---------------------------------------------------------------- A. the profiles
print("=" * 90)
print("A.  ACTUAL LOCAL ATTENTION PROFILE f(m) OF THE LATTICE HEADS")
prof = defaultdict(lambda: np.zeros(2 * SPAN + 1))
rng = np.random.default_rng(0)
with torch.no_grad():
    for i in range(0, len(X), 4):
        ch = X[i:i + 4]
        enc = task.encode([x.sequence for x in ch], [x.donor_pos for x in ch])
        A = ad.model(**enc, output_attentions=True).attentions
        S = enc["input_ids"].shape[1]
        qs = rng.integers(SPAN + 4, S - SPAN - 4, size=24)
        for (l, h) in LAT + NON:
            Al = A[l][:, h].float().cpu().numpy()
            for b in range(len(ch)):
                for p in qs:
                    prof[(l, h)] += Al[b, p, p - SPAN:p + SPAN + 1]
for k in prof:
    prof[k] /= prof[k].sum()

def describe(hd):
    f = prof[hd]
    m = np.arange(-SPAN, SPAN + 1)
    order = np.argsort(-f)
    top = [(int(m[j]), float(f[j])) for j in order[:5]]
    # participation ratio: effective number of offsets carrying the mass
    pr = 1.0 / np.sum(f ** 2)
    # mass on the diagonal (m=0)
    return top, pr, float(f[SPAN])

print(f"{'head':8s} {'strength':>8s} {'eff#offsets':>11s} {'mass@m=0':>9s}  top-5 offsets (m: mass)")
strength = {(r['layer'], r['head']): r['strength'] for r in inv['heads']}
relph = {(r['layer'], r['head']): r['dominant_rel_phase'][0] for r in inv['heads']}
for hd in sorted(LAT, key=lambda k: -strength[k])[:12]:
    top, pr, d0 = describe(hd)
    s = "  ".join(f"{m:+d}:{v:.2f}" for m, v in top)
    print(f"L{hd[0]}H{hd[1]:<5d} {strength[hd]:8.3f} {pr:11.2f} {d0:9.3f}  relphase={relph[hd]}  {s}")
print("  ... non-lattice, top-5 by strength:")
for hd in sorted(NON, key=lambda k: -strength[k])[:5]:
    top, pr, d0 = describe(hd)
    s = "  ".join(f"{m:+d}:{v:.2f}" for m, v in top)
    print(f"L{hd[0]}H{hd[1]:<5d} {strength[hd]:8.3f} {pr:11.2f} {d0:9.3f}  {s}")

# ---------------------------------------------------------------- B. the delta scan
def regions(x):
    return {"exon": (x.acceptor_pos + 6, x.donor_pos - 6),
            "intron": (x.donor_pos + 8, x.next_acceptor_pos - 8)}

def logprobs(ctx, seq_len):
    out = {"exon": [], "intron": []}
    with torch.no_grad():
        for i in range(0, len(X), 4):
            ch = X[i:i + 4]
            lp = [np.full(len(x.sequence), np.nan) for x in ch]
            for o in range(MASK_STRIDE):
                enc = ad.tokenizer([" ".join(x.sequence) for x in ch],
                                   return_tensors="pt", padding=True).to(ad.device)
                pos = []
                for j, x in enumerate(ch):
                    ps = [p for p in range(len(x.sequence)) if p % MASK_STRIDE == o]
                    for p in ps:
                        enc["input_ids"][j, p + OFF] = task.mask_id
                    pos.append(ps)
                with ctx(enc["input_ids"].shape[1] - 2):
                    lg = ad.model(**enc).logits.float()
                ll = torch.log_softmax(lg, dim=-1)
                for j, x in enumerate(ch):
                    idx = torch.tensor([p + OFF for p in pos[j]], device=ll.device)
                    tgt = torch.tensor([task.bid[x.sequence[p]] for p in pos[j]], device=ll.device)
                    vals = ll[j, idx, tgt].cpu().numpy()
                    for p, v in zip(pos[j], vals):
                        lp[j][p] = v
            for j, x in enumerate(ch):
                for name, (a, b) in regions(x).items():
                    if b - a >= 24:
                        out[name].append(float(np.nanmean(lp[j][a:b])))
    return {k: np.array(v) for k, v in out.items()}

def arm(heads):
    """damage(delta) = LL(0) - LL(delta), per-example mean, for delta = 0..6."""
    res = {}
    for d in range(7):
        def ctx(seq_len, d=d, heads=heads):
            return roll_attention(ad, heads, d, seq_len=seq_len, off=OFF)
        res[d] = logprobs(ctx, None)
    return res

def fit_b(res, region):
    base = res[0][region]
    dmg = np.array([np.mean(base - res[d][region]) for d in range(7)])
    D = np.arange(7)
    Xd = np.stack([D, (D % 3 != 0).astype(float), np.ones(7)], 1)
    coef, *_ = np.linalg.lstsq(Xd, dmg, rcond=None)
    return dmg, coef  # a, b, c

print("\n" + "=" * 90)
print("B.  THE ACTUAL DELTA-SCAN.  damage(d) = LL(intact) - LL(rolled).  fit a*d + b*1[d%3!=0] + c")
intact_ll = logprobs(lambda s: noop(), None)
print(f"intact: exon LL {intact_ll['exon'].mean():+.4f}  intron LL {intact_ll['intron'].mean():+.4f}"
      f"   chance {np.log(0.25):+.4f}")

rows = []
sets = [("LATTICE", LAT)]
rnd = random.Random(0)
for i in range(5):
    # layer-distribution-matched random draw from NON-lattice heads
    S = []
    by_l = defaultdict(list)
    for l, h in NON:
        by_l[l].append((l, h))
    ok = True
    for l, k in LAT_BY_L.items():
        if len(by_l[l]) < k:
            ok = False
            S += by_l[l]
        else:
            S += rnd.sample(by_l[l], k)
    if not ok:
        pass  # some layers have fewer than k non-lattice heads; take what exists
    sets.append((f"MATCHED-RANDOM-{i}", S))

for name, S in sets:
    res = arm(S)
    for region in ("exon", "intron"):
        dmg, (a, b, c) = fit_b(res, region)
        ll_max = res[6][region].mean()
        head = (np.log(0.25))
        hr = (ll_max - head) / (intact_ll[region].mean() - head)
        print(f"\n{name:20s} n_heads={len(S):3d}  region={region}")
        print("  damage by delta: " + "  ".join(f"d{d}={dmg[d]:+.4f}" for d in range(7)))
        print(f"  a={a:+.5f}  b={b:+.5f}  c={c:+.5f}   headroom@d=6 = {100*hr:.0f}% of intact"
              f"   (LL@6={ll_max:+.3f})")
        rows.append(dict(name=name, region=region, n=len(S), a=float(a), b=float(b),
                         c=float(c), dmg=[float(v) for v in dmg], headroom=float(hr)))

print("\n" + "=" * 90)
print("SUMMARY  (b, exon)")
for r in rows:
    if r["region"] == "exon":
        print(f"  {r['name']:20s} b = {r['b']:+.5f}")
mr = [r["b"] for r in rows if r["region"] == "exon" and r["name"].startswith("MATCHED")]
lb = [r["b"] for r in rows if r["region"] == "exon" and r["name"] == "LATTICE"][0]
print(f"\n  b_LATTICE = {lb:+.5f}   b_MR mean = {np.mean(mr):+.5f}  sd = {np.std(mr):.5f}"
      f"  range [{min(mr):+.5f}, {max(mr):+.5f}]")
(REPO / "results/phase-lattice/_audit_gate5.json").write_text(json.dumps(rows, indent=1))
