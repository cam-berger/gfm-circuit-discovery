#!/usr/bin/env python
"""AUDIT: does a mononucleotide shuffle of the exon interior floor the model at chance?

Attack claims: headroom in the shuffled region -> 0, therefore damage(delta) -> 0 for every
delta, therefore b = 0 by construction ("a control that cannot fail").

Measure: LL of the true base at masked positions in the exon interior, on
  (a) intact sequence
  (b) exon interior mononucleotide-shuffled (flanks/splice sites untouched)
and then the ACTUAL damage from rolling the 42 lattice heads by delta=1 and delta=3 in both.
"""
from __future__ import annotations
import json, random, sys
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
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40

dev = pick_device(None)
ad = load_adapter(DEFAULT_MODEL, dev)
task = CouplingTask(ad, batch=6)
OFF = task.off

inv = json.loads((REPO / "results/phase-lattice/phase_inventory.json").read_text())
LAT = [(r["layer"], r["head"]) for r in inv["heads"] if r["lattice"]]

X, _ = load_examples(DEFAULT_GTF, DEFAULT_FASTA, "val", N, task=task, tag="phasefn",
                     min_effect=None)
X = [x for x in X if x.exon_length >= MIN_LEN and x.next_acceptor_pos - x.donor_pos >= MIN_LEN]
print(f"n={len(X)} exons  lattice={len(LAT)} heads  dev={dev}")


def region(x):
    return (x.acceptor_pos + 6, x.donor_pos - 6)


rng = random.Random(0)
SEQS = {"intact": [], "shuffled": []}
for x in X:
    s = list(x.sequence)
    SEQS["intact"].append("".join(s))
    a, b = region(x)
    core = s[a:b]
    rng.shuffle(core)                      # MONONUCLEOTIDE shuffle: composition preserved
    SEQS["shuffled"].append("".join(s[:a] + core + s[b:]))


def ll_exon(seqs, delta=None):
    """mean log-prob of the TRUE base (of the sequence AS FED) in the exon interior."""
    vals = []
    with torch.no_grad():
        for i in range(0, len(X), task.batch):
            ch = X[i:i + task.batch]
            sq = seqs[i:i + task.batch]
            lp = [np.full(len(s), np.nan) for s in sq]
            for o in range(MASK_STRIDE):
                enc = ad.tokenizer([" ".join(s) for s in sq], return_tensors="pt",
                                   padding=True).to(ad.device)
                pos = []
                for j, s in enumerate(sq):
                    ps = [p for p in range(len(s)) if p % MASK_STRIDE == o]
                    for p in ps:
                        enc["input_ids"][j, p + OFF] = task.mask_id
                    pos.append(ps)
                L = max(len(s) for s in sq)
                ctx = noop if delta is None else (
                    lambda: roll_attention(ad, LAT, delta, seq_len=L, off=OFF))
                with ctx():
                    lg = ad.model(**enc).logits.float()
                ll = torch.log_softmax(lg, dim=-1)
                for j, s in enumerate(sq):
                    idx = torch.tensor([p + OFF for p in pos[j]], device=ll.device)
                    tgt = torch.tensor([task.bid[s[p]] for p in pos[j]], device=ll.device)
                    v = ll[j, idx, tgt].cpu().numpy()
                    for p, val in zip(pos[j], v):
                        lp[j][p] = val
            for j, x in enumerate(ch):
                a, b = region(x)
                seg = lp[j][a:b]
                if b - a >= 24 and not np.isnan(seg).any():
                    vals.append(float(np.mean(seg)))
    return np.array(vals)


CH = -np.log(4.0)
rows = {}
for arm in ("intact", "shuffled"):
    base = ll_exon(SEQS[arm])
    print(f"\n=== {arm.upper()} exon interior")
    print(f"  LL(no roll)   = {base.mean():+.4f}   headroom over ln(1/4) = {base.mean()-CH:+.4f} nats")
    rows[arm] = {"ll0": float(base.mean()), "headroom": float(base.mean() - CH)}
    for d in (1, 2, 3):
        r = ll_exon(SEQS[arm], d)
        dam = base.mean() - r.mean()
        print(f"  roll d={d}: LL = {r.mean():+.4f}   damage = {dam:+.4f} nats")
        rows[arm][f"ll_d{d}"] = float(r.mean())
        rows[arm][f"damage_d{d}"] = float(dam)

print("\n--- b-proxy = damage(1..2 mean) - damage(3)  [the phase term]")
for arm in rows:
    r = rows[arm]
    b = (r["damage_d1"] + r["damage_d2"]) / 2 - r["damage_d3"]
    print(f"  {arm:9s}  headroom={r['headroom']:+.4f}  damage(1)={r['damage_d1']:.4f} "
          f"damage(2)={r['damage_d2']:.4f} damage(3)={r['damage_d3']:.4f}  b~={b:+.4f}")
(REPO / "results/phase-lattice/_audit_shuffle.json").write_text(json.dumps(rows, indent=1))
