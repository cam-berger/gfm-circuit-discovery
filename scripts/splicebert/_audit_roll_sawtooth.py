"""AUDIT: does the PREREGISTERED damage metric sawtooth on a zero-information sequence?

The attack claims gates 3/4/5 are passed by attention geometry alone. Its *evidence*
was ||A'-A||_1, which is NOT the preregistered metric. The preregistered metric is
damage(d) = LL_intact - LL_rolled, LL = mean log-prob of the TRUE base at masked
positions (stride 7). So measure THAT, on sequences with no codons at all.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gfmcd.splicing.circuit import DEFAULT_MODEL, load_adapter, pick_device, roll_attention, noop

dev = pick_device(None)
ad = load_adapter(DEFAULT_MODEL, dev, eager=True)
tok, model = ad.tokenizer, ad.model
OFF = ad.prefix_tokens
MASK = tok.mask_token_id
BID = {b: tok.convert_tokens_to_ids(b) for b in "ACGT"}

inv = json.load(open(REPO / "results/phase-lattice/phase_inventory.json"))
LAT = [(r["layer"], r["head"]) for r in inv["heads"] if r["lattice"]]
NON = [(r["layer"], r["head"]) for r in inv["heads"] if not r["lattice"]]
TOPK = [(r["layer"], r["head"]) for r in inv["heads"] if r["strength"] >= 0.9999]
print(f"LAT {len(LAT)}  NON {len(NON)}  TOPK {len(TOPK)}  dev={dev}")

L = 400
rng = np.random.default_rng(0)
SEQS = {
    "ALL-A (zero info)": ["A" * L],
    "IID-RANDOM DNA (no codons)": ["".join(rng.choice(list("ACGT"), L)) for _ in range(3)],
}

STRIDE = 7
LN4 = float(np.log(0.25))


def LL(seqs, ctx):
    """mean log-prob of the true base over ALL positions, each masked exactly once."""
    vals = []
    with torch.no_grad():
        enc0 = tok([" ".join(s) for s in seqs], return_tensors="pt", padding=True).to(dev)
        S = enc0["input_ids"].shape[1]
        for o in range(STRIDE):
            enc = {k: v.clone() for k, v in enc0.items()}
            pos = [p for p in range(len(seqs[0])) if p % STRIDE == o]
            for j in range(len(seqs)):
                for p in pos:
                    enc["input_ids"][j, p + OFF] = MASK
            with ctx():
                lg = model(**enc).logits.float()
            ll = torch.log_softmax(lg, -1)
            for j, s in enumerate(seqs):
                idx = torch.tensor([p + OFF for p in pos], device=dev)
                tgt = torch.tensor([BID[s[p]] for p in pos], device=dev)
                vals += ll[j, idx, tgt].cpu().tolist()
    return float(np.mean(vals)), S


SETS = {"LATTICE-42": LAT, "NON-LATTICE-54": NON, "LATTICE-TOP-7": TOPK}
DELTAS = [0, 1, 2, 3, 4, 5, 6]

for sname, seqs in SEQS.items():
    base, S = LL(seqs, noop)
    nt = S - 2  # nucleotide columns (CLS ... SEP)
    print(f"\n=== {sname}   intact LL={base:+.4f}   headroom over ln(1/4)={base-LN4:+.4f}")
    for hname, heads in SETS.items():
        dmg, lls = [], []
        for d in DELTAS:
            ll, _ = LL(seqs, lambda hh=heads, dd=d: roll_attention(ad, hh, dd, seq_len=nt, off=OFF))
            dmg.append(base - ll)
            lls.append(ll)
        # preregistered fit: damage = a*delta + b*1[delta%3!=0] + c
        X = np.array([[d, 1.0 if d % 3 else 0.0, 1.0] for d in DELTAS])
        coef, *_ = np.linalg.lstsq(X, np.array(dmg), rcond=None)
        a, b, c = coef
        g4 = dmg[3] < dmg[2]
        g6 = dmg[6] < dmg[5]
        print(f"  {hname:15s} damage: " + " ".join(f"{v:+.4f}" for v in dmg))
        print(f"  {'':15s} LL:     " + " ".join(f"{v:+.3f}" for v in lls)
              + f"   (chance {LN4:.3f})")
        print(f"  {'':15s} FIT a={a:+.4f}  b={b:+.4f}  c={c:+.4f}   "
              f"GATE4 damage(3)<damage(2): {g4}   dip@6: {g6}")
