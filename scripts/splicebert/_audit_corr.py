"""Does self-mass a_i(0) CORRELATE with phase purity?  That is the number that decides whether
the diagonal can forge the money graph (b_norm vs purity, across 96 heads, controlling layer)."""
import json, sys
from pathlib import Path
import numpy as np

REPO = Path("/Users/cberg/Research/projects/gfm-circuit-discovery")
sm = np.load("/tmp/selfmass.npy")
inv = json.loads((REPO / "results/phase-lattice/phase_inventory.json").read_text())
STR = {(r["layer"], r["head"]): r["strength"] for r in inv["heads"]}
LAT = {(r["layer"], r["head"]) for r in inv["heads"] if r["lattice"]}

a = np.array([sm[l, h] for l in range(6) for h in range(16)])
s = np.array([STR[(l, h)] for l in range(6) for h in range(16)])
lay = np.array([l for l in range(6) for h in range(16)])
isl = np.array([(l, h) in LAT for l in range(6) for h in range(16)])

print(f"a_i(0)  lattice     mean {a[isl].mean():.4f}  median {np.median(a[isl]):.4f}  "
      f"max {a[isl].max():.4f}")
print(f"a_i(0)  non-lattice mean {a[~isl].mean():.4f}  median {np.median(a[~isl]):.4f}  "
      f"max {a[~isl].max():.4f}")
r = np.corrcoef(a, s)[0, 1]
print(f"\ncorr( a_i(0), phase strength )  over 96 heads = {r:+.4f}")
# partial correlation controlling for layer (the money graph controls for layer)
D = np.stack([np.ones(96)] + [(lay == l).astype(float) for l in range(1, 6)], 1)
ra = a - D @ np.linalg.lstsq(D, a, rcond=None)[0]
rs = s - D @ np.linalg.lstsq(D, s, rcond=None)[0]
print(f"partial corr controlling for LAYER            = {np.corrcoef(ra, rs)[0,1]:+.4f}")
print("\nper layer:")
for l in range(6):
    m = lay == l
    print(f"  L{l}: a_i(0) lattice {a[m & isl].mean() if (m&isl).any() else float('nan'):.4f}  "
          f"non-lattice {a[m & ~isl].mean() if (m&~isl).any() else float('nan'):.4f}  "
          f"corr(a,str)={np.corrcoef(a[m], s[m])[0,1]:+.3f}")
