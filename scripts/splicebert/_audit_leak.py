"""AUDIT: does the +-24 'lattice span' actually leak exon mass into intron queries?"""
import json, sys
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gfmcd.splicing.circuit import (DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask,
                                    load_adapter, load_examples, pick_device)

MIN_LEN = 60
dev = pick_device(None)
ad = load_adapter(DEFAULT_MODEL, dev, eager=True)
task = CouplingTask(ad, batch=4)
inv = json.loads((REPO / "results/phase-lattice/phase_inventory.json").read_text())
LAT = [(r["layer"], r["head"]) for r in inv["heads"] if r["lattice"]]

X, _ = load_examples(DEFAULT_GTF, DEFAULT_FASTA, "val", 80, task=task, tag="phasefn",
                     min_effect=None)
X = [x for x in X if x.exon_length >= MIN_LEN and x.next_acceptor_pos - x.donor_pos >= MIN_LEN]
print(f"{len(X)} examples")

# ---- 1. geometry: reproduce the attack's leak fractions on the real 74 -----------------
ex_tot = ex_leak = it_tot = it_leak = 0
for x in X:
    a, b = x.acceptor_pos + 6, x.donor_pos - 6
    if b - a >= 24:
        for p in range(a, b):
            ex_tot += 1
            ex_leak += (p - 24 < x.acceptor_pos or p + 24 >= x.donor_pos)
    a, b = x.donor_pos + 8, x.next_acceptor_pos - 8
    if b - a >= 24:
        for p in range(a, b):
            it_tot += 1
            it_leak += (p - 24 < x.donor_pos or p + 24 >= x.next_acceptor_pos)
print(f"GEOMETRY  intron leak {it_leak}/{it_tot} = {it_leak/it_tot:.3f} | "
      f"exon leak {ex_leak}/{ex_tot} = {ex_leak/ex_tot:.3f}")

# ---- 2. where does lattice attention MASS actually go? ---------------------------------
OFF = task.off
prof = np.zeros(2 * 200 + 1)   # local mass by offset, lattice heads, pooled
prof_n = 0
# exon-mass fraction at intron-interior queries, split leaky / clean
buckets = {("intron", "leak"): [], ("intron", "clean"): [], ("exon", "leak"): [],
           ("exon", "clean"): []}
sub = X[:24]
with torch.no_grad():
    for i in range(0, len(sub), 4):
        ch = sub[i:i + 4]
        enc = task.encode([x.sequence for x in ch], [x.donor_pos for x in ch])
        A = ad.model(**enc, output_attentions=True).attentions
        Astack = torch.stack([A[l][:, h] for (l, h) in LAT], 1).float().cpu().numpy()  # B,42,S,S
        S = enc["input_ids"].shape[1]
        for bi, x in enumerate(ch):
            L = len(x.sequence)
            exon = np.zeros(S, bool); intr = np.zeros(S, bool)
            exon[x.acceptor_pos + OFF:x.donor_pos + OFF] = True
            intr[x.donor_pos + OFF:x.next_acceptor_pos + OFF] = True
            rows = {}
            for reg, (a, b) in {"exon": (x.acceptor_pos + 6, x.donor_pos - 6),
                                "intron": (x.donor_pos + 8, x.next_acceptor_pos - 8)}.items():
                if b - a < 24:
                    continue
                lo, hi = (x.acceptor_pos, x.donor_pos) if reg == "exon" else (x.donor_pos, x.next_acceptor_pos)
                for p in range(a, b):
                    leak = "leak" if (p - 24 < lo or p + 24 >= hi) else "clean"
                    row = Astack[bi, :, p + OFF, :]          # [42, S]
                    other = exon if reg == "intron" else intr
                    buckets[(reg, leak)].append(row[:, other].sum(1))
                    if reg == "intron" and leak == "clean":
                        pass
            # offset profile from mid-sequence queries
            for p in range(200, L - 200, 37):
                row = Astack[bi, :, p + OFF, :]
                prof += row[:, p + OFF - 200:p + OFF + 201].sum(0)
                prof_n += len(LAT)

prof /= prof_n
k = np.arange(-200, 201)
loc = prof.copy(); loc[200] = 0    # exclude self
tot_row = 1.0
print("\nLATTICE-HEAD ATTENTION MASS BY OFFSET (mean over 42 heads, per row; row sums to 1)")
print(f"  self (k=0)          : {prof[200]:.4f}")
for lim in (3, 6, 12, 24, 48, 100, 200):
    m = loc[np.abs(k) <= lim].sum()
    print(f"  |k| <= {lim:3d} (non-self): {m:.4f}")
print(f"  beyond |k|>200 or specials: {1 - prof.sum():.4f}")

print("\nEXON-MASS FRACTION carried by lattice heads (mean over 42 heads, per query row):")
for key, v in buckets.items():
    if not v:
        continue
    V = np.stack(v)   # [nq, 42]
    print(f"  {key[0]:6s} {key[1]:5s} n={len(v):5d}  mass on the OTHER region: "
          f"mean {V.mean():.4f}  median {np.median(V.mean(1)):.4f}  "
          f"p90 {np.percentile(V.mean(1), 90):.4f}")
