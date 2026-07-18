#!/usr/bin/env python
"""IC1 pilot v3 (ACC_UP=70): calibrate the CONSERVATIVE-SCRAMBLE primary arm + a pilot-frozen
DETERMINISTIC sham (population damage-matched, NOT per-intron outcome-matched -> no collider).

Conservative scramble = anchored dinuc shuffle with cryptic-site changes BOUNDED BY CONSTRUCTION
(rejection): |Δmax score3|<=band3, |Δmax score5|<=band5, |Δmax polyY|<=bandY, no NEW 23-mer with
score3>strong_thr, and count(score3>strong_thr) preserved -- plus the dose floor. Reports YIELD
(can it make N_SEED valid shuffles without collapsing dose?), dose, OOD, signed+dispersion MDE.
Sham sweep: fixed PPT purine-swap rules of increasing strength; pick the one whose median f-shift
best matches the partner corruption A- at the POPULATION level; freeze it deterministically.
Disjoint TRAIN split; linear f.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts" / "splicebert"))

from gfmcd.provenance import ExperimentManifest, sha256_file                      # noqa: E402
from gfmcd.splicing.circuit import (                                             # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, load_adapter, pick_device, eligible, _dataset)
from gfmcd.splicing.coupling import substitute                                    # noqa: E402
from run_ic1_pilot import anchored_shuffle, f_linear, window, hamming, ACC_UPS    # noqa: E402
from run_ic1_pilot2 import max_score3, max_score5, max_polyY, HAVE5               # noqa: E402
from maxentpy import maxent                                                       # noqa: E402
from maxentpy.maxent import load_matrix3                                          # noqa: E402

M3 = load_matrix3()
ACC_UP = 70
STRONG_THR = 6.0            # a "competitive" cryptic 3'SS score (bits)
BAND3 = BAND5 = 2.0
BANDY = 2


def count_strong(s):
    return sum(1 for i in range(len(s) - 22)
              if set(s[i:i + 23].upper()) <= set("ACGT") and maxent.score3(s[i:i + 23].upper(), matrix=M3) > STRONG_THR)


def conservative_scramble(W, rng, attempts=400):
    b3, b5, bY = max_score3(W), max_score5(W), max_polyY(W)
    bc = count_strong(W)
    for _ in range(attempts):
        c = anchored_shuffle(W, rng)
        if hamming(W, c) < 8 or hamming(W, c) / len(W) < 0.25:
            continue
        if abs(max_score3(c) - b3) > BAND3:
            continue
        if HAVE5 and abs(max_score5(c) - b5) > BAND5:
            continue
        if abs(max_polyY(c) - bY) > BANDY:
            continue
        if max_score3(c) > max(b3, STRONG_THR) + 1e-9:
            continue
        if count_strong(c) != bc:
            continue
        return c
    return None


SHAM_RULES = {"k2_-10": (10, "AA"), "k3_-11": (11, "AAA"), "k4_-12": (12, "AAAA"), "k4_-8": (8, "AAAA")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pilot", type=int, default=80)
    ap.add_argument("--n-seed", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF); ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="results/splicebert_coupling/ic1_pilot3.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    dev = pick_device(args.device); ad = load_adapter(args.model, dev); task = CouplingTask(ad, batch=args.batch)

    train = eligible(_dataset(args.gtf, args.fasta), "train")
    seen, dedup = set(), []
    for x in train:
        if x.gene_id not in seen:
            seen.add(x.gene_id); dedup.append(x)

    def elig(x):
        lo, hi = window(x, ACC_UP)
        return 0 <= lo and hi < len(x.sequence) and (hi - lo + 1) >= 30

    pool = [x for x in dedup if elig(x)]

    # ---- conservative-scramble YIELD + dose (over a scan of the pool) ----
    yielded, cons_pilot = 0, []
    scan = 0
    for x in pool:
        lo, hi = window(x, ACC_UP); W = x.sequence[lo:hi + 1]
        # dose pre-check (same as other arms)
        pre = [anchored_shuffle(W, np.random.default_rng((hash(x.gene_id) % 99991) * 31 + s)) for s in range(args.n_seed)]
        if not (np.median([hamming(W, s) for s in pre]) >= 8
                and np.median([hamming(W, s) for s in pre]) / len(W) >= 0.25 and len(set(pre)) == args.n_seed):
            continue
        scan += 1
        cons = []
        for s in range(args.n_seed):
            c = conservative_scramble(W, np.random.default_rng((hash(x.gene_id) % 90001) * 17 + s + 1))
            if c is None:
                break
            cons.append(c)
        if len(cons) == args.n_seed and len(set(cons)) == args.n_seed:
            yielded += 1
            if len(cons_pilot) < args.n_pilot:
                cons_pilot.append((x, lo, hi, W, cons))
        if scan >= 600 and len(cons_pilot) >= args.n_pilot:
            break
    cons_yield = yielded / max(scan, 1)
    ham_cons = [np.median([hamming(p[3], c) for c in p[4]]) for p in cons_pilot]
    print(f"pool={len(pool)} scanned(dose-ok)={scan} conservative-yield={cons_yield:.2f} n_cons_pilot={len(cons_pilot)}")

    # ---- sham sweep (native only): median f-shift ratio vs A- ----
    seqs, dons, tags = [], [], []
    for gi, (x, lo, hi, W, cons) in enumerate(cons_pilot):
        a, d0 = x.next_acceptor_pos, x.donor_pos
        seqs += [x.sequence, substitute(x.sequence, a - 2, "CC")]; dons += [d0, d0]
        tags += [(gi, "A+"), (gi, "A-")]
        for name, (off, alt) in SHAM_RULES.items():
            seqs.append(substitute(x.sequence, a - off, alt)); dons.append(d0); tags.append((gi, name))
    fv = f_linear(ad, task, seqs, dons, args.batch)
    bysh = {}
    for (gi, k), v in zip(tags, fv):
        bysh.setdefault(gi, {})[k] = v
    sham_sweep = {}
    for name in SHAM_RULES:
        ratios = [abs(bysh[gi][name] - bysh[gi]["A+"]) / (abs(bysh[gi]["A-"] - bysh[gi]["A+"]) + 1e-9)
                  for gi in range(len(cons_pilot))]
        sham_sweep[name] = {"fshift_ratio_med": float(np.median(ratios)),
                            "fshift_ratio_iqr": [float(np.percentile(ratios, 25)), float(np.percentile(ratios, 75))]}
    sham_pick = min(SHAM_RULES, key=lambda n: abs(sham_sweep[n]["fshift_ratio_med"] - 1.0))
    off_pick, alt_pick = SHAM_RULES[sham_pick]
    print(f"sham sweep: " + "  ".join(f"{n}={sham_sweep[n]['fshift_ratio_med']:.2f}" for n in SHAM_RULES)
          + f"  -> PICK {sham_pick}")

    # ---- conservative-arm ΔC, dispersion vs frozen sham, signed+dispersion MDE ----
    seqs, dons, tags = [], [], []
    for gi, (x, lo, hi, W, cons) in enumerate(cons_pilot):
        a, d0 = x.next_acceptor_pos, x.donor_pos
        Am = substitute(x.sequence, a - 2, "CC"); Ash = substitute(x.sequence, a - off_pick, alt_pick)
        seqs += [x.sequence, Am, Ash]; dons += [d0, d0, d0]; tags += [(gi, "I", "+"), (gi, "I", "-"), (gi, "I", "s")]
        for s, c in enumerate(cons):
            sp = x.sequence[:lo] + c + x.sequence[hi + 1:]
            seqs += [sp, substitute(sp, a - 2, "CC"), substitute(sp, a - off_pick, alt_pick)]
            dons += [d0, d0, d0]; tags += [(gi, s, "+"), (gi, s, "-"), (gi, s, "s")]
    fv = f_linear(ad, task, seqs, dons, args.batch)
    by = {}
    for (gi, s, w), v in zip(tags, fv):
        by.setdefault(gi, {})[(s, w)] = v
    dcbar, disp, disp_sham, excess, dc_sham, ood_check = [], [], [], [], [], []
    for gi in range(len(cons_pilot)):
        b = by[gi]; base = b[("I", "+")] - b[("I", "-")]; base_s = b[("I", "+")] - b[("I", "s")]
        dcs = [(b[(s, "+")] - b[(s, "-")]) - base for s in range(args.n_seed)]
        dss = [(b[(s, "+")] - b[(s, "s")]) - base_s for s in range(args.n_seed)]
        dcbar.append(float(np.mean(dcs))); dc_sham.append(float(np.mean(dss)))
        disp.append(float(np.std(dcs))); disp_sham.append(float(np.std(dss)))
        excess.append(float(np.std(dcs) - np.std(dss)))
    zc = 2.8016; n_elig_cons = int(round(cons_yield * len(pool)))
    sd_between = float(np.std(dcbar, ddof=1)); sd_excess = float(np.std(excess, ddof=1))
    # verify cryptic constraint held by construction (should be all within band)
    d3max = max(abs(max_score3(c) - max_score3(p[3])) for p in cons_pilot for c in p[4])

    rep = {"acc_up": ACC_UP, "n_seed": args.n_seed, "strong_thr": STRONG_THR, "bands": [BAND3, BAND5, BANDY],
           "conservative": {"yield": cons_yield, "n_eligible_est": n_elig_cons, "n_pilot": len(cons_pilot),
                            "dose_norm_hamming_med": float(np.median(ham_cons)) / float(np.median([len(p[3]) for p in cons_pilot])),
                            "dose_altered_med": float(np.median(ham_cons)),
                            "max_score3_change_observed": float(d3max), "constraint_ok": bool(d3max <= BAND3 + 1e-6),
                            "dCbar_absmean": float(np.mean(np.abs(dcbar))), "sd_between": sd_between,
                            "signed_mde_census": float(zc * sd_between / np.sqrt(max(n_elig_cons, 1))),
                            "disp_shuffle_mean": float(np.mean(disp)), "disp_sham_mean": float(np.mean(disp_sham)),
                            "excess_mean": float(np.mean(excess)), "excess_sd_between": sd_excess,
                            "disp_mde_census": float(zc * sd_excess / np.sqrt(max(n_elig_cons, 1)))},
           "sham_sweep": sham_sweep, "sham_pick": {"rule": sham_pick, "offset": off_pick, "alt": alt_pick,
                            "fshift_ratio_med": sham_sweep[sham_pick]["fshift_ratio_med"],
                            "dC_sham_mean": float(np.mean(dc_sham)), "dC_sham_p95abs": float(np.percentile(np.abs(dc_sham), 95))}}
    out = REPO / args.out; out.write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))
    ckpt = Path(args.model); ckpt = ckpt if ckpt.is_absolute() else REPO / ckpt
    man = ExperimentManifest.create("splicebert_coupling:ic1_pilot3", sys.argv, status="PILOT",
        config={"acc_up": ACC_UP, "n_seed": args.n_seed, "strong_thr": STRONG_THR}, seeds={"base": 0},
        notes=["IC1 pilot v3: conservative-scramble arm + deterministic damage-matched sham."])
    man.add_input(str(ckpt / "pytorch_model.bin")); man.write(out.parent / "ic1_pilot3.manifest.json")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
