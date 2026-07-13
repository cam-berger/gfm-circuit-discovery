"""AUDIT part 2: the preregistered damage metric on REAL hg38, exon interior vs intron interior."""
import json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gfmcd.splicing.circuit import (DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask,
                                    load_adapter, load_examples, noop, pick_device, roll_attention)

dev = pick_device(None)
ad = load_adapter(DEFAULT_MODEL, dev, eager=True)
task = CouplingTask(ad, batch=6)
OFF, MASK, BID = task.off, task.mask_id, task.bid

inv = json.load(open(REPO / "results/phase-lattice/phase_inventory.json"))
LAT = [(r["layer"], r["head"]) for r in inv["heads"] if r["lattice"]]
NON = [(r["layer"], r["head"]) for r in inv["heads"] if not r["lattice"]]

X, _ = load_examples(DEFAULT_GTF, DEFAULT_FASTA, "val", 6, task=task, tag="phasefn",
                     min_effect=None)
X = [x for x in X if x.exon_length >= 60 and x.next_acceptor_pos - x.donor_pos >= 60]
print(f"{len(X)} examples")
STRIDE, LN4 = 7, float(np.log(0.25))


def regions(x):
    return {"exon": (x.acceptor_pos + 6, x.donor_pos - 6),
            "intron": (x.donor_pos + 8, x.next_acceptor_pos - 8)}


def LL(ctx):
    acc = {"exon": [], "intron": []}
    with torch.no_grad():
        for i in range(0, len(X), task.batch):
            ch = X[i:i + task.batch]
            lp = [np.full(len(x.sequence), np.nan) for x in ch]
            enc0 = ad.tokenizer([" ".join(x.sequence) for x in ch], return_tensors="pt",
                                padding=True).to(dev)
            nt = enc0["input_ids"].shape[1] - 2
            for o in range(STRIDE):
                enc = {k: v.clone() for k, v in enc0.items()}
                pos = []
                for j, x in enumerate(ch):
                    ps = [p for p in range(len(x.sequence)) if p % STRIDE == o]
                    for p in ps:
                        enc["input_ids"][j, p + OFF] = MASK
                    pos.append(ps)
                with ctx(nt):
                    ll = torch.log_softmax(ad.model(**enc).logits.float(), -1)
                for j, x in enumerate(ch):
                    idx = torch.tensor([p + OFF for p in pos[j]], device=dev)
                    tgt = torch.tensor([BID[x.sequence[p]] for p in pos[j]], device=dev)
                    for p, v in zip(pos[j], ll[j, idx, tgt].cpu().numpy()):
                        lp[j][p] = v
            for j, x in enumerate(ch):
                for name, (a, b) in regions(x).items():
                    acc[name] += list(lp[j][a:b])
    return {k: float(np.nanmean(v)) for k, v in acc.items()}


base = LL(lambda nt: noop())
print(f"INTACT  exon {base['exon']:+.4f} (headroom {base['exon']-LN4:+.4f})   "
      f"intron {base['intron']:+.4f} (headroom {base['intron']-LN4:+.4f})   chance {LN4:.4f}")

DELTAS = [0, 1, 2, 3, 4, 5, 6]
Xd = np.array([[d, 1.0 if d % 3 else 0.0, 1.0] for d in DELTAS])
for hname, heads in (("LATTICE-42", LAT), ("NON-LATTICE-54", NON)):
    res = {"exon": [], "intron": []}
    hr = []
    for d in DELTAS:
        r = LL(lambda nt, hh=heads, dd=d: roll_attention(ad, hh, dd, seq_len=nt, off=OFF))
        for k in res:
            res[k].append(base[k] - r[k])
        hr.append(r["exon"] - LN4)
    print(f"\n{hname}")
    bs = {}
    for k in ("exon", "intron"):
        dmg = np.array(res[k])
        a, b, c = np.linalg.lstsq(Xd, dmg, rcond=None)[0]
        bs[k] = b
        print(f"  {k:7s} damage: " + " ".join(f"{v:+.4f}" for v in dmg)
              + f"   a={a:+.4f} b={b:+.4f}  GATE4 d(3)<d(2): {dmg[3] < dmg[2]}  dip@6: {dmg[6] < dmg[5]}")
    print(f"  DiD b_exon - b_intron = {bs['exon'] - bs['intron']:+.4f}")
    print(f"  health (exon headroom / intact {base['exon']-LN4:.3f}): "
          + " ".join(f"{h/(base['exon']-LN4):.2f}" for h in hr))
