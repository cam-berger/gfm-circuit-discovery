"""a_i(0) at MASKED query positions, all 96 heads. No genome load: the attention PATTERN is
content-blind (it survives sequence shuffling -- run_phase_inventory.py docstring), so random
ACGT with the same 7-stride masking gives the magnitude. This is the number run_phase_inventory
never measured (it does `if k == 0: continue` at line 92-93)."""
import json, sys
from pathlib import Path
import numpy as np, torch

REPO = Path("/Users/cberg/Research/projects/gfm-circuit-discovery")
sys.path.insert(0, str(REPO / "src"))
from gfmcd.splicing.circuit import DEFAULT_MODEL, CouplingTask, load_adapter, pick_device

MASK_STRIDE, SPAN, N, LEN = 7, 24, 6, 1024
dev = pick_device(None)
ad = load_adapter(DEFAULT_MODEL, dev, eager=True)
task = CouplingTask(ad, batch=2)
OFF = task.off
inv = json.loads((REPO / "results/phase-lattice/phase_inventory.json").read_text())
STR = {(r["layer"], r["head"]): r["strength"] for r in inv["heads"]}
LAT = {(r["layer"], r["head"]) for r in inv["heads"] if r["lattice"]}

rng = np.random.default_rng(0)
seqs = ["".join(rng.choice(list("ACGT"), LEN)) for _ in range(N)]

self_m = torch.zeros(6, 16, dtype=torch.float64)
local_m = torch.zeros(6, 16, dtype=torch.float64)
n = 0
with torch.no_grad():
    for i in range(0, N, 2):
        ch = seqs[i:i + 2]
        for o in range(MASK_STRIDE):
            enc = ad.tokenizer([" ".join(s) for s in ch], return_tensors="pt",
                               padding=True).to(ad.device)
            for j in range(len(ch)):
                ps = torch.tensor([p + OFF for p in range(LEN) if p % MASK_STRIDE == o],
                                  device=ad.device)
                enc["input_ids"][j, ps] = task.mask_id
            A = ad.model(**enc, output_attentions=True).attentions
            for j in range(len(ch)):
                q = torch.tensor([p + OFF for p in range(SPAN + 4, LEN - SPAN - 4)
                                  if p % MASK_STRIDE == o], device=ad.device)
                cols = q[:, None] + torch.arange(-SPAN, SPAN + 1, device=ad.device)[None, :]
                ar = torch.arange(len(q), device=ad.device)[:, None]
                for l in range(6):
                    win = A[l][j].float()[:, q][:, ar, cols]        # [H,Q,2*SPAN+1]
                    s = win[:, :, SPAN]
                    self_m[l] += s.sum(1).cpu().double()
                    local_m[l] += (win.sum(-1) - s).sum(1).cpu().double()
                n += len(q)
        print(f"  {min(i+2,N)}/{N}", end="\r", flush=True)
print()

sm, lm = (self_m / n).numpy(), (local_m / n).numpy()
np.save("/tmp/selfmass.npy", sm)
a = np.array([sm[l, h] for l in range(6) for h in range(16)])
s = np.array([STR[(l, h)] for l in range(6) for h in range(16)])
lay = np.array([l for l in range(6) for h in range(16)])
isl = np.array([(l, h) in LAT for l in range(6) for h in range(16)])

print(f"a_i(0) at a MASKED query, averaged over {n} scored positions\n")
print(f"  LATTICE (42):     mean {a[isl].mean():.4f}  median {np.median(a[isl]):.4f}  "
      f"min {a[isl].min():.4f}  max {a[isl].max():.4f}")
print(f"  NON-LATTICE (54): mean {a[~isl].mean():.4f}  median {np.median(a[~isl]):.4f}  "
      f"min {a[~isl].min():.4f}  max {a[~isl].max():.4f}")
print(f"\n  corr(a_i(0), phase strength) over 96 heads          = "
      f"{np.corrcoef(a, s)[0,1]:+.4f}")
D = np.stack([np.ones(96)] + [(lay == l).astype(float) for l in range(1, 6)], 1)
ra = a - D @ np.linalg.lstsq(D, a, rcond=None)[0]
rs = s - D @ np.linalg.lstsq(D, s, rcond=None)[0]
print(f"  partial corr, controlling for LAYER (the money graph) = "
      f"{np.corrcoef(ra, rs)[0,1]:+.4f}")

print("\nper layer  (mean a_i(0)):")
for l in range(6):
    m = lay == l
    la = a[m & isl].mean() if (m & isl).any() else float("nan")
    na = a[m & ~isl].mean() if (m & ~isl).any() else float("nan")
    print(f"  L{l}:  lattice {la:.4f}   non-lattice {na:.4f}   "
          f"corr(a,strength) within layer = {np.corrcoef(a[m], s[m])[0,1]:+.3f}")

print("\nLAYER-0 lattice heads (where the self column really IS the bare [MASK] embedding):")
for (l, h) in sorted(LAT):
    if l == 0:
        fr = sm[l, h] / (sm[l, h] + lm[l, h])
        print(f"  L0H{h:<3d} a_i(0)={sm[l,h]:.4f}  local off-diag={lm[l,h]:.4f}  "
              f"self/(self+local)={fr:.4f}  strength={STR[(l,h)]:.3f}")
print("\nDONE")
