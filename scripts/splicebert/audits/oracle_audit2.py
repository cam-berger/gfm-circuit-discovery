import json, pickle, sys
import numpy as np
sys.path.insert(0, "/Users/cberg/Research/projects/test/GFM_SpliceBERT_Circuit_Discovery_Implementation/gfm_splicebert_implementation/src")
from gfmcd.splicing.io import load_jsonl
from gfmcd.splicing.baselines import LocalPairBaseline, site_window
from gfmcd.splicing.schema import SplicePairExample

ROOT = "/Users/cberg/Research/projects/test/GFM_SpliceBERT_Circuit_Discovery_Implementation/gfm_splicebert_implementation"
with open(f"{ROOT}/colab_results/splicebert_run/baselines/local_pair_baseline.pkl", "rb") as f:
    bl: LocalPairBaseline = pickle.load(f)
conf = load_jsonl(f"{ROOT}/colab_results/splicebert_run/baselines/val_local_conflicts.jsonl", validate=False)

# 1. true acceptor position across the FULL val split (sample) and train (sample)
for name in ("val", "train"):
    poss, n = set(), 0
    with open(f"{ROOT}/data/processed/splicebert_partner_selection/v1/{name}.jsonl") as f:
        for line in f:
            x = SplicePairExample.from_json(line)
            poss.add(x.pairs[x.true_pair_index].acceptor_pos)
            n += 1
            if n >= 300: break
    print(f"{name}: first {n} examples -> distinct true acceptor positions = {sorted(poss)}")

# 2. baseline PMR on its own conflict set (circularity check)
m = np.asarray([bl.true_margin(x) for x in conf])
print("\nPWM+length baseline on the conflict set: PMR = %.4f, mean margin %.3f, max margin %.3f"
      % ((m > 0).mean(), m.mean(), m.max()))

# 3. full-set half oracles
def rule_pmr(acc_rule, don_rule, use_len=True, M=10.0):
    ok = 0
    for x in conf:
        tp = x.pairs[x.true_pair_index]
        s = []
        for p in x.pairs:
            if acc_rule == "pos200":
                sa = M * (p.acceptor_pos == 200)
            elif acc_rule == "oracle":
                sa = M * (p.acceptor_pos == tp.acceptor_pos)
            else:
                sa = bl.acceptor_pwm.score(site_window(x.sequence, p.acceptor_pos, "acceptor"))
            if don_rule == "oracle":
                sd = M * (p.donor_pos == tp.donor_pos)
            else:
                sd = bl.donor_pwm.score(site_window(x.sequence, p.donor_pos, "donor"))
            v = sa + sd
            if use_len:
                v += bl.length_weight * bl.length_prior.score(p.exon_length)
            s.append(v)
        s = np.asarray(s)
        ok += (s[x.true_pair_index] - np.delete(s, x.true_pair_index).max()) > 0
    return ok / len(conf)

print("\n--- rival scorers on the FULL n=%d conflict set (all strictly ADDITIVE, zero partner info) ---" % len(conf))
print("PWM acc + PWM don + length            (= the registered baseline) : %.4f" % rule_pmr("pwm", "pwm"))
print("POSITION-200 acc + PWM don + length   (knows only 'acceptor@200') : %.4f" % rule_pmr("pos200", "pwm"))
print("PWM acc + ORACLE don + length                                     : %.4f" % rule_pmr("pwm", "oracle"))
print("ORACLE acc + ORACLE don + length      (oracle site detector)      : %.4f" % rule_pmr("oracle", "oracle"))
print("ORACLE acc + ORACLE don, NO length    (pure site detection)       : %.4f" % rule_pmr("oracle", "oracle", use_len=False, M=1.0))
print("\nobserved additive-frozen conflict PMR = 0.807 ; additive-finetuned = 0.997")

# 4. length == donor position, given acceptor is pinned at 200
lens = [x.pairs[x.true_pair_index].exon_length for x in conf]
dons = [x.pairs[x.true_pair_index].donor_pos for x in conf]
print("\ncorr(true exon_length, true donor_pos) over conflicts = %.6f  (identity: donor = 200 + length)"
      % np.corrcoef(lens, dons)[0, 1])
