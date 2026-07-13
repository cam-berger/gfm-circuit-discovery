"""AUDIT: measure a_i(0) -- self-attention mass at MASKED query positions -- for all 96 heads,
under the exact 7-stride masking regime the phase-shift design will use.
run_phase_inventory.py:92-93 does `if k == 0: continue`, so this has never been measured."""
import json, sys
from pathlib import Path
import numpy as np, torch

REPO = Path("/Users/cberg/Research/projects/gfm-circuit-discovery")
sys.path.insert(0, str(REPO / "src"))
from gfmcd.splicing.circuit import (DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask,
                                    load_adapter, load_examples, pick_device)

MASK_STRIDE, SPAN = 7, 24
dev = pick_device(None)
ad = load_adapter(DEFAULT_MODEL, dev, eager=True)
task = CouplingTask(ad, batch=2)
OFF = task.off
inv = json.loads((REPO / "results/phase-lattice/phase_inventory.json").read_text())
LATSET = {(r["layer"], r["head"]) for r in inv["heads"] if r["lattice"]}
STR = {(r["layer"], r["head"]): r["strength"] for r in inv["heads"]}

X, _ = load_examples(DEFAULT_GTF, DEFAULT_FASTA, "val", 8, task=task, tag="audit-self",
                     min_effect=None)
print(f"{len(X)} seqs  device={dev}", flush=True)

self_m = torch.zeros(6, 16, dtype=torch.float64)   # a_i(0) at masked query
local_m = torch.zeros(6, 16, dtype=torch.float64)  # sum_{0<|k|<=SPAN} a_i(k) at masked query
n = 0
with torch.no_grad():
    for i in range(0, len(X), 2):
        ch = X[i:i + 2]
        for o in range(MASK_STRIDE):
            enc = ad.tokenizer([" ".join(x.sequence) for x in ch], return_tensors="pt",
                               padding=True).to(ad.device)
            for j, x in enumerate(ch):
                ps = torch.tensor([p + OFF for p in range(len(x.sequence))
                                   if p % MASK_STRIDE == o], device=ad.device)
                enc["input_ids"][j, ps] = task.mask_id
            A = ad.model(**enc, output_attentions=True).attentions
            for j, x in enumerate(ch):
                lo, hi = SPAN + 4, len(x.sequence) - SPAN - 4
                q = torch.tensor([p + OFF for p in range(lo, hi) if p % MASK_STRIDE == o],
                                 device=ad.device)
                cols = q[:, None] + torch.arange(-SPAN, SPAN + 1, device=ad.device)[None, :]
                for l in range(6):
                    R = A[l][j].float()                       # [H,S,S]
                    win = R[:, q][:, torch.arange(len(q))[:, None], cols]  # [H,Q,2S+1]
                    s = win[:, :, SPAN]                        # k == 0
                    tot = win.sum(-1) - s
                    self_m[l] += s.sum(1).double().cpu()
                    local_m[l] += tot.sum(1).double().cpu()
                n += len(q)
        print(f"  {min(i+2,len(X))}/{len(X)}", end="\r", flush=True)
print()

sm = (self_m / n).numpy()
lm = (local_m / n).numpy()
lat = np.array([[sm[l, h], lm[l, h], STR[(l, h)]] for (l, h) in sorted(LATSET)])
non = np.array([[sm[l, h], lm[l, h]] for l in range(6) for h in range(16)
                if (l, h) not in LATSET])

print(f"\n{'head':9s} {'phase-str':>9s} {'a_i(0)':>9s} {'local off-diag':>15s} {'self frac':>10s}")
for (l, h) in sorted(LATSET):
    fr = sm[l, h] / max(sm[l, h] + lm[l, h], 1e-12)
    print(f"L{l}H{h:<6d} {STR[(l,h)]:9.3f} {sm[l,h]:9.4f} {lm[l,h]:15.4f} {fr:10.4f}")

print("\n" + "=" * 76)
print(f"LATTICE (n={len(lat)}):     a_i(0) mean {lat[:,0].mean():.4f}  median "
      f"{np.median(lat[:,0]):.4f}  max {lat[:,0].max():.4f}  min {lat[:,0].min():.4f}")
print(f"NON-LATTICE (n={len(non)}): a_i(0) mean {non[:,0].mean():.4f}  median "
      f"{np.median(non[:,0]):.4f}  max {non[:,0].max():.4f}")
print(f"SUM of a_i(0) over the 42 lattice heads = {lat[:,0].sum():.3f} head-units of mass "
      f"that the roll moves off [MASK] onto a real base")
print(f"mean self/(self+local) frac, lattice = {(lat[:,0]/(lat[:,0]+lat[:,1])).mean():.4f}")
print("\nlargest-self lattice heads:")
for l, h in sorted(LATSET, key=lambda t: -sm[t[0], t[1]])[:8]:
    print(f"  L{l}H{h:<3d} a_i(0)={sm[l,h]:.4f} local={lm[l,h]:.4f} str={STR[(l,h)]:.3f}")
print("\nall 96 heads")
np.save("/tmp/selfmass.npy", sm)
