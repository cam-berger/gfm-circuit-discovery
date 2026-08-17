"""Item 1 -- audit the 96-head scan: scale, uncertainty, and the missing null.

The scan reported point estimates for 96 heads with NO interval on any of them, NO specificity
control (it ran with --n-random-sets 0), and no accounting for having looked at 96 heads. It
also compared FRACTIONAL drops, and the two arms differ 4.7x in baseline, so the choice of
scale is itself a finding that needs defending rather than assuming.

POST-OUTCOME NOTICE. The first scan landed before any decision rule was frozen. Everything
here is therefore a DESCRIPTIVE audit with honest multiplicity accounting -- not a
confirmatory test, and no threshold below is preregistered. Read every p-value as a
descriptive statistic.

Design. Pairs are matched 1:1 and 100% cross-gene, so the pair is the exchangeability unit:

  CI    bootstrap over PAIRS (each pair one draw)
  NULL  permute the class label WITHIN each pair independently; recompute all 96 heads;
        take the max |asymmetry| across heads per permutation. That max-statistic null is
        what turns 96 look-heres into one family-wise threshold.

Scale. Reported on BOTH the fractional and absolute scales, plus the residual above a fitted
proportional relationship -- which is the scale-free version of the question.
"""

import json
import sys

import numpy as np

SCRATCH = ("/private/tmp/claude-501/-Users-cberg-Research-projects-gfm-circuit-discovery--"
           "claude-worktrees-pr-audit-neurips-integration-630dfb/"
           "fee7d82b-0cbb-4d7c-95d4-7610592d772e/scratchpad")
SRC = f"{SCRATCH}/m2m3_allheads_rows.json"
OUT = f"{SCRATCH}/item1_scan_audit.json"
N_BOOT, N_PERM, SEED = 2000, 2000, 0

d = json.load(open(SRC))
ph = d["per_head"]
br = d["baseline_rows"]
if br is None:
    sys.exit("scan artifact has no per-row dump; re-run with --dump-rows")

bt = np.array(br["treatment_Y"], float)
bc = np.array(br["comparator_Y"], float)
n = len(bt)
assert len(bc) == n, "arms are not 1:1 paired"
heads = [k for k in ph if "rows" in ph[k]]
layer = {k: int(k.split("H")[0][1:]) for k in heads}
print(f"pairs={n}  heads with per-row data={len(heads)}")

YT = np.array([ph[k]["rows"]["treatment_Y"] for k in heads])   # [H, n]
YC = np.array([ph[k]["rows"]["comparator_Y"] for k in heads])


def stats(sel_t):
    """Fractional drops for both arms given a boolean per-pair 'treatment stays treatment'.

    Where sel_t[i] is False the pair's two rows swap arms, which is the null's exchange.
    """
    base_t = np.where(sel_t, bt, bc)
    base_c = np.where(sel_t, bc, bt)
    abl_t = np.where(sel_t[None, :], YT, YC)
    abl_c = np.where(sel_t[None, :], YC, YT)
    mt, mc = base_t.mean(), base_c.mean()
    ft = (mt - abl_t.mean(axis=1)) / mt
    fc = (mc - abl_c.mean(axis=1)) / mc
    return ft, fc


keep = np.ones(n, dtype=bool)
ft0, fc0 = stats(keep)
asym0 = ft0 - fc0
abs_t0 = bt.mean() - YT.mean(axis=1)
abs_c0 = bc.mean() - YC.mean(axis=1)

# ---------------------------------------------------------------- CI over pairs
rng = np.random.default_rng(SEED)
boot = np.empty((N_BOOT, len(heads)))
for b in range(N_BOOT):
    idx = rng.integers(0, n, n)
    mt, mc = bt[idx].mean(), bc[idx].mean()
    boot[b] = ((mt - YT[:, idx].mean(axis=1)) / mt) - ((mc - YC[:, idx].mean(axis=1)) / mc)
lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)

# ------------------------------------------------- max-statistic permutation null
rng2 = np.random.default_rng(SEED + 1)
maxnull = np.empty(N_PERM)
for p in range(N_PERM):
    sel = rng2.random(n) < 0.5
    a, c = stats(sel)
    maxnull[p] = np.max(np.abs(a - c))
crit = float(np.quantile(maxnull, 0.95))
fw_p = np.array([(1 + np.sum(maxnull >= abs(a))) / (N_PERM + 1) for a in asym0])

# ------------------------------------- scale-free: residual above proportional fit
slope = float((abs_t0 * abs_c0).sum() / (abs_c0 * abs_c0).sum())
resid = abs_t0 - slope * abs_c0

print()
print("=" * 78)
print("Family-wise threshold from the max-statistic permutation null")
print("=" * 78)
print(f"  95th pct of max|asymmetry| under class-label exchange: {crit:.4f}")
print(f"  heads whose |asymmetry| exceeds it: "
      f"{int(np.sum(np.abs(asym0) > crit))} of {len(heads)}")
print()
print(f"  proportional fit slope {slope:.4f}  (equal-fractional null predicts "
      f"{bt.mean()/bc.mean():.4f}; equal-absolute null predicts 1.0)")

order = np.argsort(-asym0)
print()
print("=" * 78)
print("Heads surviving the family-wise threshold, by asymmetry")
print("=" * 78)
print(f"{'head':>7} {'lyr':>3} {'frac asym':>10} {'95% CI':>20} {'fw p':>8} "
      f"{'abs asym':>9} {'resid':>8}")
survivors = []
for i in order:
    k = heads[i]
    star = "*" if abs(asym0[i]) > crit else " "
    if star == "*":
        survivors.append(k)
    if star == "*" or i in order[:6]:
        print(f"{k:>7} {layer[k]:>3} {asym0[i]:>+10.4f} "
              f"[{lo[i]:>+.4f},{hi[i]:>+.4f}] {fw_p[i]:>8.4f} "
              f"{abs_t0[i]-abs_c0[i]:>+9.4f} {resid[i]:>+8.4f} {star}")

print()
print("=" * 78)
print("Layer summary, all three scales")
print("=" * 78)
print(f"{'lyr':>3} {'n':>3} {'med frac asym':>14} {'med abs asym':>13} {'med resid':>10} "
      f"{'n survive':>10}")
for L in range(6):
    idx = [i for i, k in enumerate(heads) if layer[k] == L]
    ns = sum(1 for i in idx if abs(asym0[i]) > crit)
    print(f"{L:>3} {len(idx):>3} {np.median(asym0[idx]):>+14.4f} "
          f"{np.median(abs_t0[idx]-abs_c0[idx]):>+13.4f} {np.median(resid[idx]):>+10.4f} "
          f"{ns:>10}")

# ------------------------------------------- is layer 0 elevated as a BLOCK?
l0 = np.array([i for i, k in enumerate(heads) if layer[k] == 0])
rest = np.array([i for i, k in enumerate(heads) if layer[k] != 0])
obs_block = float(np.median(asym0[l0]) - np.median(asym0[rest]))
rng3 = np.random.default_rng(SEED + 2)
blocknull = np.empty(N_PERM)
for p in range(N_PERM):
    sel = rng3.random(n) < 0.5
    a, c = stats(sel)
    s = a - c
    blocknull[p] = np.median(s[l0]) - np.median(s[rest])
block_p = float((1 + np.sum(np.abs(blocknull) >= abs(obs_block))) / (N_PERM + 1))
print()
print("=" * 78)
print("Layer-0-as-a-block contrast (median asymmetry, L0 vs all other layers)")
print("=" * 78)
print(f"  observed {obs_block:+.4f}   permutation p = {block_p:.4f}")
print("  This is ONE prespecifiable contrast, so it carries no 96-head multiplicity --")
print("  but it was formed after seeing the first scan, so it is post-outcome.")

json.dump({
    "schema": "gfmcd.item1.scan_audit/1",
    "post_outcome": True,
    "note": "descriptive audit; no threshold here was preregistered",
    "n_pairs": n, "n_heads": len(heads),
    "familywise_critical_value": crit,
    "n_surviving": len(survivors), "survivors": survivors,
    "proportional_slope": slope,
    "equal_fractional_null_slope": float(bt.mean() / bc.mean()),
    "layer0_block_contrast": {"observed": obs_block, "permutation_p": block_p},
    "per_head": {heads[i]: {"layer": layer[heads[i]], "frac_asymmetry": float(asym0[i]),
                            "ci": [float(lo[i]), float(hi[i])],
                            "familywise_p": float(fw_p[i]),
                            "abs_asymmetry": float(abs_t0[i] - abs_c0[i]),
                            "proportional_residual": float(resid[i])}
                 for i in range(len(heads))},
}, open(OUT, "w"), indent=2)
print(f"\nwrote {OUT}")
