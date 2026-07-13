"""Is the registered PWM a strawman? Refit with matched (+-12nt) windows and a tuned length weight."""
import json, pickle, sys, itertools
import numpy as np
sys.path.insert(0, "/Users/cberg/Research/projects/test/GFM_SpliceBERT_Circuit_Discovery_Implementation/gfm_splicebert_implementation/src")
from gfmcd.splicing.baselines import PositionWeightModel, LengthPrior, LocalPairBaseline

ROOT = "/Users/cberg/Research/projects/test/GFM_SpliceBERT_Circuit_Discovery_Implementation/gfm_splicebert_implementation"
D = f"{ROOT}/data/processed/splicebert_partner_selection/v1"

def win(seq, pos, up, down):
    s, e = pos - up, pos + down
    if s < 0 or e > len(seq): return None
    return seq[s:e]

def stream(path, n):
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n: break
            d = json.loads(line)
            out.append(d)
    return out

NTR, NVA = 25000, 4000
tr = stream(f"{D}/train.jsonl", NTR)
va = stream(f"{D}/val.jsonl", NVA)
print("train sample", len(tr), "val sample", len(va))

# registered baseline (fit on FULL train, from the run)
with open(f"{ROOT}/colab_results/splicebert_run/baselines/local_pair_baseline.pkl", "rb") as f:
    reg: LocalPairBaseline = pickle.load(f)

def top1(acc_up, acc_dn, don_up, don_dn, lw, acc_pwm, don_pwm, lp, data):
    ok = 0
    for d in data:
        seq = d["sequence"]; ti = d["true_pair_index"]
        best, bi = -1e18, -1
        for j, p in enumerate(d["pairs"]):
            a = acc_pwm.score(win(seq, p["acceptor_pos"], acc_up, acc_dn))
            b = don_pwm.score(win(seq, p["donor_pos"], don_up, don_dn))
            L = p["donor_pos"] - p["acceptor_pos"]
            s = a + b + lw * lp.score(L)
            if s > best: best, bi = s, j
        ok += (bi == ti)
    return ok / len(data)

# sanity: reproduce registered baseline on the val sample
reg_top1 = 0
for d in va:
    seq = d["sequence"]; ti = d["true_pair_index"]
    sc = [reg.acceptor_pwm.score(win(seq, p["acceptor_pos"], 20, 3)) +
          reg.donor_pwm.score(win(seq, p["donor_pos"], 3, 6)) +
          reg.length_weight * reg.length_prior.score(p["donor_pos"] - p["acceptor_pos"]) for p in d["pairs"]]
    reg_top1 += int(np.argmax(sc) == ti)
print("registered baseline (acc -20..+3, don -3..+6, lw=1.0)  val-sample top1 = %.4f" % (reg_top1/len(va)))

lengths = [d["pairs"][d["true_pair_index"]]["donor_pos"] - d["pairs"][d["true_pair_index"]]["acceptor_pos"] for d in tr]
lp = LengthPrior.fit(lengths, min_length=min(lengths), max_length=max(lengths))

for (au, ad, du, dd) in [(20, 3, 3, 6), (12, 12, 12, 12), (40, 10, 10, 12)]:
    aw = [w for d in tr if (w := win(d["sequence"], d["pairs"][d["true_pair_index"]]["acceptor_pos"], au, ad))]
    dw = [w for d in tr if (w := win(d["sequence"], d["pairs"][d["true_pair_index"]]["donor_pos"], du, dd))]
    A = PositionWeightModel.fit(aw); B = PositionWeightModel.fit(dw)
    for lw in (0.0, 0.5, 1.0, 2.0, 3.0):
        t = top1(au, ad, du, dd, lw, A, B, lp, va)
        print("  acc(-%d..+%d) don(-%d..+%d) lw=%.1f  val-sample top1 = %.4f" % (au, ad, du, dd, lw, t))
