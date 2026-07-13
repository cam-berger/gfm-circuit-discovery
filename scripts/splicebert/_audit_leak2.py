"""AUDIT 2: coding mass = BOTH flanking exons.  How much coding value-mass does a lattice
head actually collect at an intron-interior query, leaky vs clean?  And what would a 24nt
(or 48nt) erosion actually buy?"""
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
NON = [(l, h) for l in range(ad.num_layers) for h in range(ad.num_heads)
       if (l, h) not in set(LAT)]

X, _ = load_examples(DEFAULT_GTF, DEFAULT_FASTA, "val", 80, task=task, tag="phasefn",
                     min_effect=None)
X = [x for x in X if x.exon_length >= MIN_LEN and x.next_acceptor_pos - x.donor_pos >= MIN_LEN]
OFF = task.off
sub = X[:24]

# coding = upstream exon [acc,don) UNION downstream exon [next_acc, next_acc+len) -- we do not
# know the downstream exon end, so use the whole tail (an UPPER bound on coding mass).
res = {}
with torch.no_grad():
    for i in range(0, len(sub), 4):
        ch = sub[i:i + 4]
        enc = task.encode([x.sequence for x in ch], [x.donor_pos for x in ch])
        A = ad.model(**enc, output_attentions=True).attentions
        Ast = torch.stack([A[l][:, h] for (l, h) in LAT], 1).float().cpu().numpy()
        S = enc["input_ids"].shape[1]
        for bi, x in enumerate(ch):
            L = len(x.sequence)
            coding = np.zeros(S, bool)
            coding[x.acceptor_pos + OFF:x.donor_pos + OFF] = True
            coding[x.next_acceptor_pos + OFF:L + OFF] = True     # upper bound
            a, b = x.donor_pos + 8, x.next_acceptor_pos - 8
            if b - a < 24:
                continue
            lo, hi = x.donor_pos, x.next_acceptor_pos
            for p in range(a, b):
                d = min(p - lo, hi - 1 - p)          # distance to nearest splice boundary
                row = Ast[bi, :, p + OFF, :]         # [42, S]
                m = row[:, coding].sum(1).mean()     # coding mass, mean over 42 lattice heads
                res.setdefault("all", []).append((d, m))

D = np.array([d for d, _ in res["all"]])
M = np.array([m for _, m in res["all"]])
print(f"intron-interior queries scored: {len(D)}")
print("\nCODING mass (both flanking exons; UPPER bound) collected by the 42 lattice heads,")
print("as a function of the query's distance d to the nearest splice boundary:")
for lo, hi in [(0, 6), (6, 12), (12, 24), (24, 48), (48, 999)]:
    s = (D >= lo) & (D < hi)
    if s.sum():
        print(f"  {lo:3d} <= d < {hi:3d}  n={s.sum():5d}   coding mass = {M[s].mean():.4f}")
print(f"\n  ALL intron-interior (current design, erosion 8)      : {M.mean():.4f}")
for e in (24, 48):
    s = D >= e
    print(f"  after the proposed erosion of {e:2d}nt  n={s.sum():5d} ({s.mean():.1%} kept)"
          f"  : {M[s].mean() if s.sum() else float('nan'):.4f}")
print("\n=> the erosion removes only the difference between these numbers.")
