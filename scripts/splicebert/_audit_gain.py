"""AUDIT: is there a region-level SCALE GAIN s_exon/s_intron?

The attack: damage_region(d) = s_region * g(d).  Then b_exon - b_intron = b*(s_e - s_i) > 0
with IDENTICAL phase shape.  DiD (a difference) cannot tell that from a codon effect.

Decisive test: fit a,b,c PER REGION on the real roll intervention and compare the RATIOS
a_e/a_i, c_e/c_i, b_e/b_i.  Pure gain <=> all three ratios equal.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, load_adapter,
    load_examples, noop, pick_device, roll_attention,
)

MASK_STRIDE = 7
MIN_LEN = 60
N = 24
DELTAS = [0, 1, 2, 3, 4, 5, 6]

dev = pick_device(None)
ad = load_adapter(DEFAULT_MODEL, dev, eager=True)
task = CouplingTask(ad, batch=4)
OFF = task.off

inv = json.loads((REPO / "results/phase-lattice/phase_inventory.json").read_text())
LAT = [(r["layer"], r["head"]) for r in inv["heads"] if r["lattice"]]
print(f"lattice heads {len(LAT)}  device {dev}")

X, _ = load_examples(DEFAULT_GTF, DEFAULT_FASTA, "val", N, task=task, tag="audit",
                     min_effect=None)
X = [x for x in X if x.exon_length >= MIN_LEN
     and x.next_acceptor_pos - x.donor_pos >= MIN_LEN]
L = {len(x.sequence) for x in X}
print(f"{len(X)} examples, seq lengths {sorted(L)[:3]}...")
# uniform length so the nucleotide column range is unambiguous (no PAD inside it)
mode_len = max(L, key=lambda l: sum(len(x.sequence) == l for x in X))
X = [x for x in X if len(x.sequence) == mode_len]
print(f"{len(X)} examples at length {mode_len}")

def regions(x):
    return {"exon": (x.acceptor_pos + 6, x.donor_pos - 6),
            "intron": (x.donor_pos + 8, x.next_acceptor_pos - 8)}

def profile_means(ctx):
    """mean log-prob of the true base, per example, per region."""
    out = {"exon": [], "intron": []}
    with torch.no_grad():
        for i in range(0, len(X), task.batch):
            ch = X[i:i + task.batch]
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
                with ctx():
                    lg = ad.model(**enc).logits.float()
                ll = torch.log_softmax(lg, dim=-1)
                for j, x in enumerate(ch):
                    idx = torch.tensor([p + OFF for p in pos[j]], device=ll.device)
                    tgt = torch.tensor([task.bid[x.sequence[p]] for p in pos[j]],
                                       device=ll.device)
                    for p, v in zip(pos[j], ll[j, idx, tgt].cpu().numpy()):
                        lp[j][p] = v
            for j, x in enumerate(ch):
                for name, (a, b) in regions(x).items():
                    if b - a >= 24:
                        out[name].append(float(np.nanmean(lp[j][a:b])))
    return {k: np.array(v) for k, v in out.items()}

base = profile_means(noop)
print("\nINTACT   exon %.4f   intron %.4f" % (base["exon"].mean(), base["intron"].mean()))
CH = -np.log(4)
print("HEADROOM exon %.4f   intron %.4f" % (base["exon"].mean()-CH, base["intron"].mean()-CH))

dmg = {"exon": {}, "intron": {}}
for d in DELTAS:
    ctx = (lambda d=d: roll_attention(ad, LAT, d, seq_len=mode_len, off=OFF))
    m = profile_means(ctx)
    for r in ("exon", "intron"):
        dmg[r][d] = base[r] - m[r]          # per-example damage (nats)
    print("delta=%d  damage exon %+.4f   intron %+.4f   | absLL exon %.4f intron %.4f"
          % (d, dmg["exon"][d].mean(), dmg["intron"][d].mean(),
             m["exon"].mean(), m["intron"].mean()))

# fit damage(d) = a*d + b*1[d%3!=0] + c  per region
D = np.array(DELTAS, float)
Mx = np.stack([D, (D % 3 != 0).astype(float), np.ones_like(D)], 1)
print("\n%-8s %8s %8s %8s" % ("region", "a", "b", "c"))
coef = {}
for r in ("exon", "intron"):
    y = np.array([dmg[r][d].mean() for d in DELTAS])
    a, b, c = np.linalg.lstsq(Mx, y, rcond=None)[0]
    coef[r] = (a, b, c)
    print("%-8s %8.5f %8.5f %8.5f" % (r, a, b, c))

ae, be, ce = coef["exon"]; ai, bi, ci = coef["intron"]
print("\nDiD = b_exon - b_intron = %+.5f" % (be - bi))
print("\nTHE ATTACK'S TEST -- pure gain <=> these three ratios are EQUAL:")
print("  a_exon/a_intron = %7.3f" % (ae / ai))
print("  b_exon/b_intron = %7.3f" % (be / bi))
print("  c_exon/c_intron = %7.3f" % (ce / ci))
print("\nSCALE-FREE contrast b/a:   exon %.4f   intron %.4f" % (be/ae, bi/ai))
np.save("/tmp/audit_dmg.npy", np.array([[dmg[r][d] for d in DELTAS] for r in ("exon","intron")]))
