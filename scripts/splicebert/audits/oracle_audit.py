"""Audit: what PMR does an ORACLE SITE DETECTOR get on the conflict set?"""
import json, pickle, sys, math
from collections import Counter
import numpy as np

sys.path.insert(0, "/Users/cberg/Research/projects/test/GFM_SpliceBERT_Circuit_Discovery_Implementation/gfm_splicebert_implementation/src")
from gfmcd.splicing.io import load_jsonl
from gfmcd.splicing.baselines import LocalPairBaseline, site_window

ROOT = "/Users/cberg/Research/projects/test/GFM_SpliceBERT_Circuit_Discovery_Implementation/gfm_splicebert_implementation"
conf = load_jsonl(f"{ROOT}/colab_results/splicebert_run/baselines/val_local_conflicts.jsonl", validate=False)
print("n conflict examples:", len(conf))
with open(f"{ROOT}/colab_results/splicebert_run/baselines/local_pair_baseline.pkl", "rb") as f:
    bl: LocalPairBaseline = pickle.load(f)
print("length_weight:", bl.length_weight, "prior range:", bl.length_prior.min_length, bl.length_prior.max_length)
lp = bl.length_prior.log_prob
print("length prior log_prob: min %.3f max %.3f span %.3f" % (lp.min(), lp.max(), lp.max()-lp.min()))

# ---- structural facts
true_acc_pos = Counter()
n_pairs = []
n_acc = []; n_don = []
share_a = []; share_d = []; share_both = []; share_none = []
for x in conf:
    tp = x.pairs[x.true_pair_index]
    true_acc_pos[tp.acceptor_pos] += 1
    n_pairs.append(len(x.pairs)); n_acc.append(len(x.acceptors)); n_don.append(len(x.donors))
    sa = sd = sb = sn = 0
    for i, p in enumerate(x.pairs):
        if i == x.true_pair_index: continue
        a = p.acceptor_pos == tp.acceptor_pos
        d = p.donor_pos == tp.donor_pos
        if a and d: sb += 1
        elif a: sa += 1
        elif d: sd += 1
        else: sn += 1
    share_a.append(sa); share_d.append(sd); share_both.append(sb); share_none.append(sn)

print("\n--- structure ---")
print("distinct true acceptor positions:", true_acc_pos.most_common(5))
print("mean #pairs %.1f  mean #acceptors %.1f  mean #donors %.1f" % (np.mean(n_pairs), np.mean(n_acc), np.mean(n_don)))
print("decoys sharing TRUE ACCEPTOR (wrong donor): mean %.1f" % np.mean(share_a))
print("decoys sharing TRUE DONOR (wrong acceptor): mean %.1f" % np.mean(share_d))
print("decoys sharing BOTH true endpoints: total %d" % sum(share_both))
print("decoys sharing NEITHER: mean %.1f" % np.mean(share_none))

# ---- oracle site detector: perfect true-site score M, spurious motifs 0, + length prior
def oracle_pmr(M, use_length=True):
    ok = 0
    worst_margin = math.inf
    for x in conf:
        tp = x.pairs[x.true_pair_index]
        scores = []
        for p in x.pairs:
            s = M * ((p.acceptor_pos == tp.acceptor_pos) + (p.donor_pos == tp.donor_pos))
            if use_length:
                s += bl.length_weight * bl.length_prior.score(p.exon_length)
            scores.append(s)
        scores = np.asarray(scores)
        t = scores[x.true_pair_index]
        d = np.delete(scores, x.true_pair_index).max()
        m = t - d
        worst_margin = min(worst_margin, m)
        ok += m > 0
    return ok / len(conf), worst_margin

for M in (0.5, 1.0, 2.0, 5.0, 10.0):
    pmr, wm = oracle_pmr(M, use_length=True)
    print("ORACLE site detector (margin M=%.1f) + length prior : conflict PMR = %.4f  (worst margin %.3f)" % (M, pmr, wm))
pmr, wm = oracle_pmr(1.0, use_length=False)
print("ORACLE site detector (M=1), NO length prior at all      : conflict PMR = %.4f  (worst margin %.3f)" % (pmr, wm))

# ---- half oracles: perfect acceptor detector + real PWM donor, and vice versa
def half_oracle(perfect: str, M=10.0):
    ok = 0
    for x in conf:
        tp = x.pairs[x.true_pair_index]
        scores = []
        for p in x.pairs:
            aw = site_window(x.sequence, p.acceptor_pos, "acceptor")
            dw = site_window(x.sequence, p.donor_pos, "donor")
            if perfect == "acceptor":
                sa = M * (p.acceptor_pos == tp.acceptor_pos)
                sd = bl.donor_pwm.score(dw)
            else:
                sa = bl.acceptor_pwm.score(aw)
                sd = M * (p.donor_pos == tp.donor_pos)
            scores.append(sa + sd + bl.length_weight * bl.length_prior.score(p.exon_length))
        scores = np.asarray(scores)
        t = scores[x.true_pair_index]
        d = np.delete(scores, x.true_pair_index).max()
        ok += (t - d) > 0
    return ok / len(conf)

sub = conf[:2000]
_all = conf
conf = sub
print("\n--- half oracles (first 2000 conflicts) ---")
print("perfect ACCEPTOR detector + PWM donor + length : conflict PMR = %.4f" % half_oracle("acceptor"))
print("perfect DONOR detector + PWM acceptor + length : conflict PMR = %.4f" % half_oracle("donor"))
conf = _all

# ---- what does the baseline get WRONG: acceptor, donor, or both?
kinds = Counter()
for x in conf:
    tp = x.pairs[x.true_pair_index]
    scores = bl.score_example(x)
    j = int(np.nanargmax(scores))
    p = x.pairs[j]
    a_ok = p.acceptor_pos == tp.acceptor_pos
    d_ok = p.donor_pos == tp.donor_pos
    kinds[("acc_ok" if a_ok else "acc_WRONG", "don_ok" if d_ok else "don_WRONG")] += 1
print("\n--- what the PWM baseline's argmax gets wrong on its own conflict set ---")
tot = sum(kinds.values())
for k, v in kinds.most_common():
    print("  %-24s %6d  (%.1f%%)" % (str(k), v, 100*v/tot))
