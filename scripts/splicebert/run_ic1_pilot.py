#!/usr/bin/env python
"""IC1 pre-freeze pilot: anchored-shuffler unit tests + disjoint (train-split) census/MDE/dose/OOD
calibration across ACC_UP in {45,70,100}. Freezes IC1's numeric thresholds. NOT the confirmatory run.

Protocol: studies/splicebert_partner_selection/PREREGISTRATION_IC1_DISTAL_INTRON_NECESSITY.md
- Linear observable f = logit(G) - mean(logit A,C,T) at the masked donor (NOT the max-margin task metric).
- Distal window W = [d0+7 .. a-ACC_UP-1]; d0=donor_pos (GT at d0), a=next_acceptor_pos (AG at [a-2,a-1]).
- PRIMARY perturbation = anchored dinucleotide shuffle (whole-window dinuc preserved; internal GT/AG
  positions held fixed; interstitial runs Altschul-Erikson shuffled). SECONDARY = unconstrained dinuc shuffle.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, Counter
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest, git_commit, sha256_file          # noqa: E402
from gfmcd.splicing.circuit import (                                             # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, load_adapter, pick_device,
    eligible, _dataset,
)
from gfmcd.splicing.coupling import substitute                                    # noqa: E402

ACC_UPS = [45, 70, 100]
DON_DN = 6            # distal window starts at d0 + DON_DN + 1
MIN_DISTAL = 30


# ============================================================ dinucleotide shuffles
def dinuc_shuffle(seq, rng):
    """Altschul-Erikson: preserve dinucleotide counts + first/last char. Returns seq if unshufflable."""
    n = len(seq)
    if n <= 2:
        return seq
    first, last = seq[0], seq[-1]
    edges = defaultdict(list)
    for i in range(n - 1):
        edges[seq[i]].append(seq[i + 1])
    verts = sorted(set(seq))    # DETERMINISM FIX (v8): set-of-chars iteration is PYTHONHASHSEED-dependent
                                # -> would vary the Eulerian vertex order across processes -> non-reproducible
                                # shuffles/cohort (255-261). sorted() makes it hashseed-independent.
    last_edge = None
    for _ in range(60):
        cand, ok = {}, True
        for v in verts:
            if v == last:
                continue
            outs = edges[v]
            if not outs:
                return seq
            cand[v] = outs[int(rng.integers(len(outs)))]
        good = True
        for v in verts:
            if v == last:
                continue
            seen, cur = set(), v
            while cur != last:
                if cur in seen or cur not in cand:
                    good = False
                    break
                seen.add(cur)
                cur = cand[cur]
            if not good:
                break
        if good:
            last_edge = cand
            break
    if last_edge is None:
        return seq
    order = {}
    for v in verts:
        outs = list(edges[v])
        if v != last:
            outs.remove(last_edge[v])
        outs = [outs[i] for i in rng.permutation(len(outs))]
        if v != last:
            outs.append(last_edge[v])
        order[v] = outs
    res, cur, idx = [first], first, defaultdict(int)
    for _ in range(n - 1):
        nxt = order[cur][idx[cur]]
        idx[cur] += 1
        res.append(nxt)
        cur = nxt
    return "".join(res)


def anchor_mask(W):
    m = [False] * len(W)
    for i in range(len(W) - 1):
        if W[i:i + 2] in ("GT", "AG"):
            m[i] = m[i + 1] = True
    return m


def anchored_shuffle(W, rng):
    """Whole-window dinuc preserved; internal GT/AG positions FIXED; interstitial runs dinuc-shuffled."""
    m = anchor_mask(W)
    res = list(W)
    i, n = 0, len(W)
    while i < n:
        if m[i]:
            i += 1
            continue
        j = i
        while j < n and not m[j]:
            j += 1
        run = W[i:j]
        if len(run) >= 3:
            res[i:j] = list(dinuc_shuffle(run, rng))
        i = j
    return "".join(res)


def unconstrained_shuffle(W, rng):
    return dinuc_shuffle(W, rng)


def dinuc_counts(s):
    return Counter(s[i:i + 2] for i in range(len(s) - 1))


def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


# ============================================================ unit tests (no model)
def run_unit_tests():
    rng = np.random.default_rng(0)
    fails = []
    bases = "ACGT"
    cases = ["".join(bases[k] for k in rng.integers(0, 4, L)) for L in (20, 35, 60, 90)]
    cases += ["AGTAGTAGTCCCTTTGGGAGAG", "GTGTGTAGAGAGCACACACACA", "AAAA", "ACGTACGT"]
    for W in cases:
        for fn, name in ((anchored_shuffle, "anchored"), (unconstrained_shuffle, "unconstrained")):
            sh = fn(W, rng)
            if len(sh) != len(W):
                fails.append(f"{name}: length changed on {W}")
            if dinuc_counts(sh) != dinuc_counts(W):
                fails.append(f"{name}: dinuc multiset changed on {W}")
        # anchored MUST preserve GT/AG anchor positions
        sh = anchored_shuffle(W, rng)
        if anchor_mask(sh) != anchor_mask(W):
            fails.append(f"anchored: GT/AG anchor positions moved on {W}")
    # identity: unshufflable window returns input (a run with <3 nt or single-symbol)
    if dinuc_shuffle("AA", rng) != "AA":
        fails.append("dinuc_shuffle short-window not identity")
    # characterize realized shuffle-space size on a typical 60nt window
    W = "".join(bases[k] for k in rng.integers(0, 4, 60))
    seen = {anchored_shuffle(W, rng) for _ in range(200)}
    space = len(seen)
    return {"PASS": len(fails) == 0, "failures": fails, "anchored_space_60nt_of_200": space}


# ============================================================ model forwards
def f_linear(ad, task, seqs, donors, batch):
    """linear donor margin f = logit(G) - mean(logit A,C,T) at masked donor."""
    off, bid = task.off, task.bid
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            ss, dd = seqs[i:i + batch], donors[i:i + batch]
            enc = task.encode(ss, dd)
            lg = ad.model(**enc).logits.float()
            for j, d in enumerate(dd):
                v = lg[j, d + off]
                out.append(float(v[bid["G"]] - (v[bid["A"]] + v[bid["C"]] + v[bid["T"]]) / 3))
    return np.array(out)


def ood_nll(ad, task, seqs, wins, batch):
    """mean masked-LM NLL over the (all-masked) distal window W; OOD proxy in nats."""
    off, tok = task.off, ad.tokenizer
    bid = task.bid
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            ss, ww = seqs[i:i + batch], wins[i:i + batch]
            enc = tok([" ".join(s) for s in ss], return_tensors="pt", padding=True).to(ad.device)
            for j, (lo, hi) in enumerate(ww):
                enc["input_ids"][j, lo + off:hi + off + 1] = task.mask_id
            lp = ad.model(**enc).logits.float().log_softmax(-1)
            for j, (lo, hi) in enumerate(ww):
                nll = [-float(lp[j, p + off, bid[ss[j][p]]]) for p in range(lo, hi + 1)
                       if ss[j][p] in bid]
                out.append(float(np.mean(nll)) if nll else float("nan"))
    return np.array(out)


# ============================================================ pilot
def window(x, acc_up):
    lo = x.donor_pos + DON_DN + 1
    hi = x.next_acceptor_pos - acc_up - 1
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit-only", action="store_true")
    ap.add_argument("--n-pilot", type=int, default=80)
    ap.add_argument("--n-seed", type=int, default=8)
    ap.add_argument("--n-seed-hi", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF); ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="results/splicebert_coupling/ic1_pilot.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    unit = run_unit_tests()
    print(f"UNIT TESTS: PASS={unit['PASS']}  anchored_space(60nt,200 draws)={unit['anchored_space_60nt_of_200']}")
    for f in unit["failures"]:
        print("  XX", f)
    if args.unit_only:
        return
    if not unit["PASS"]:
        print("*** unit tests FAILED -- not running pilot"); return

    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev)
    task = CouplingTask(ad, batch=args.batch)

    # disjoint pilot set: TRAIN split, deduped to one intron per gene
    train = eligible(_dataset(args.gtf, args.fasta), "train")
    seen_g, dedup = set(), []
    for x in train:
        if x.gene_id in seen_g:
            continue
        seen_g.add(x.gene_id); dedup.append(x)
    print(f"train eligible introns={len(train)}  unique-gene deduped={len(dedup)}")

    zcrit = 1.959964 + 0.8416212       # two-sided alpha=0.05, power=0.80 -> 2.8016
    rng = np.random.default_rng(0)
    report = {}
    for acc_up in ACC_UPS:
        elig = [x for x in dedup
                if 0 <= window(x, acc_up)[0] and window(x, acc_up)[1] < len(x.sequence)
                and (window(x, acc_up)[1] - window(x, acc_up)[0] + 1) >= MIN_DISTAL]
        n_elig = len(elig)
        pilot = elig[:args.n_pilot]
        # ---- ΔC via linear f + dose, primary (anchored) arm ----
        seqsC, donsC, tag = [], [], []          # (which, gene_idx, seed)
        native_pairs = []                        # (I,A+),(I,A-) per pilot intron
        dose_rows, seedhi_rows = [], []
        for gi, x in enumerate(pilot):
            lo, hi = window(x, acc_up); W = x.sequence[lo:hi + 1]
            # native f(I,A+), f(I,A-)
            seqsC.append(x.sequence); donsC.append(x.donor_pos); tag.append(("I+", gi, -1))
            seqsC.append(substitute(x.sequence, x.next_acceptor_pos - 2, "CC"))
            donsC.append(x.donor_pos); tag.append(("I-", gi, -1))
            shuffles = []
            for s in range(args.n_seed):
                rs = np.random.default_rng((gi + 1) * 100003 + s)
                Ws = anchored_shuffle(W, rs); shuffles.append(Ws)
                seqp = x.sequence[:lo] + Ws + x.sequence[hi + 1:]
                seqsC.append(seqp); donsC.append(x.donor_pos); tag.append(("Ip+", gi, s))
                seqsC.append(substitute(seqp, x.next_acceptor_pos - 2, "CC"))
                donsC.append(x.donor_pos); tag.append(("Ip-", gi, s))
            distinct = len(set(shuffles))
            ham = [hamming(W, Ws) for Ws in shuffles]
            dose_rows.append({"gene": x.gene_id, "ldist": len(W), "distinct": distinct,
                              "altered_med": float(np.median(ham)),
                              "norm_hamming_med": float(np.median(ham)) / max(len(W), 1)})
        fvals = f_linear(ad, task, seqsC, donsC, args.batch)
        by = defaultdict(dict)
        for (which, gi, s), fv in zip(tag, fvals):
            by[gi][(which, s)] = fv
        dCbar, dCvar = [], []
        for gi in range(len(pilot)):
            d = by[gi]
            base = d[("I+", -1)] - d[("I-", -1)]
            dcs = [(d[("Ip+", s)] - d[("Ip-", s)]) - base for s in range(args.n_seed)]
            dCbar.append(float(np.mean(dcs))); dCvar.append(float(np.var(dcs)))
        sd_between = float(np.std(dCbar, ddof=1)) if len(dCbar) > 1 else float("nan")
        mde_census = zcrit * sd_between / np.sqrt(max(n_elig, 1))
        mde_pilot = zcrit * sd_between / np.sqrt(max(len(pilot), 1))
        # ---- seed-stability: dispersion estimate at n_seed vs n_seed_hi on a subset ----
        sub = pilot[:min(30, len(pilot))]
        disp8, disp16 = [], []
        s2, d2, t2 = [], [], []
        for gi, x in enumerate(sub):
            lo, hi = window(x, acc_up); W = x.sequence[lo:hi + 1]
            s2.append(x.sequence); d2.append(x.donor_pos); t2.append((gi, -1))
            s2.append(substitute(x.sequence, x.next_acceptor_pos - 2, "CC")); d2.append(x.donor_pos); t2.append((gi, -2))
            for s in range(args.n_seed_hi):
                rs = np.random.default_rng((gi + 1) * 777 + s)
                Ws = anchored_shuffle(W, rs); seqp = x.sequence[:lo] + Ws + x.sequence[hi + 1:]
                s2.append(seqp); d2.append(x.donor_pos); t2.append((gi, ("p+", s)))
                s2.append(substitute(seqp, x.next_acceptor_pos - 2, "CC")); d2.append(x.donor_pos); t2.append((gi, ("p-", s)))
        f2 = f_linear(ad, task, s2, d2, args.batch)
        b2 = defaultdict(dict)
        for (gi, k), fv in zip(t2, f2):
            b2[gi][k] = fv
        for gi in range(len(sub)):
            d = b2[gi]; base = d[-1] - d[-2]
            dcs = [(d[("p+", s)] - d[("p-", s)]) - base for s in range(args.n_seed_hi)]
            disp8.append(float(np.std(dcs[:args.n_seed]))); disp16.append(float(np.std(dcs)))
        # ---- OOD band: distal-window NLL, natural vs shuffled, 4 cells (subset) ----
        oodsub = pilot[:min(50, len(pilot))]
        seqsO, winsO, cellO = [], [], []
        for x in oodsub:
            lo, hi = window(x, acc_up); W = x.sequence[lo:hi + 1]
            seqsO.append(x.sequence); winsO.append((lo, hi)); cellO.append("I+")
            seqsO.append(substitute(x.sequence, x.next_acceptor_pos - 2, "CC")); winsO.append((lo, hi)); cellO.append("I-")
            for s in range(3):
                rs = np.random.default_rng((hash(x.gene_id) % 9973) * 13 + s)
                Ws = anchored_shuffle(W, rs); seqp = x.sequence[:lo] + Ws + x.sequence[hi + 1:]
                seqsO.append(seqp); winsO.append((lo, hi)); cellO.append("Ip+")
                seqsO.append(substitute(seqp, x.next_acceptor_pos - 2, "CC")); winsO.append((lo, hi)); cellO.append("Ip-")
        ood = ood_nll(ad, task, seqsO, winsO, args.batch)
        cells = defaultdict(list)
        for c, v in zip(cellO, ood):
            if not np.isnan(v):
                cells[c].append(v)
        nat = np.array(cells["I+"] + cells["I-"])
        band_hi = float(np.percentile(nat, 95)) if len(nat) else float("nan")
        shuf = np.array(cells["Ip+"] + cells["Ip-"])
        frac_in_band = float(np.mean(shuf <= band_hi)) if len(shuf) else float("nan")

        report[str(acc_up)] = {
            "n_eligible_genes": n_elig, "n_pilot": len(pilot),
            "sd_between_dCbar": sd_between, "mde_census": float(mde_census), "mde_pilot": float(mde_pilot),
            "meets_mde_0.05": bool(mde_census <= 0.05), "meets_n80": bool(n_elig >= 80),
            "dCbar_mean": float(np.mean(dCbar)), "dCbar_absmean": float(np.mean(np.abs(dCbar))),
            "dispersion_mean": float(np.mean(np.sqrt(dCvar))),
            "dose": {"distinct_med": float(np.median([d["distinct"] for d in dose_rows])),
                     "norm_hamming_med": float(np.median([d["norm_hamming_med"] for d in dose_rows])),
                     "altered_med": float(np.median([d["altered_med"] for d in dose_rows])),
                     "frac_ge8_altered_and_h0.25": float(np.mean(
                         [(d["altered_med"] >= 8 and d["norm_hamming_med"] >= 0.25) for d in dose_rows]))},
            "seed_stability": {"disp_sd_at8_vs16_corr": float(np.corrcoef(disp8, disp16)[0, 1])
                               if len(disp8) > 2 else None,
                               "disp_mean8": float(np.mean(disp8)), "disp_mean16": float(np.mean(disp16))},
            "ood": {"natural_nll_med": float(np.median(nat)) if len(nat) else None,
                    "band_hi_p95": band_hi, "shuffled_nll_med": float(np.median(shuf)) if len(shuf) else None,
                    "frac_shuffled_in_band": frac_in_band},
        }
        print(f"ACC_UP={acc_up}: n_elig={n_elig} sd_between={sd_between:.4f} "
              f"MDE_census={mde_census:.4f} (<=0.05: {mde_census<=0.05}) n>=80: {n_elig>=80} "
              f"| dose distinct={report[str(acc_up)]['dose']['distinct_med']:.0f} "
              f"normHam={report[str(acc_up)]['dose']['norm_hamming_med']:.2f} "
              f"frac_ok={report[str(acc_up)]['dose']['frac_ge8_altered_and_h0.25']:.2f} "
              f"| OOD in-band={frac_in_band:.2f}")

    out = REPO / args.out
    payload = {"unit_tests": unit, "n_seed": args.n_seed, "min_distal": MIN_DISTAL,
               "zcrit_a05_pw80": zcrit, "per_acc_up": report,
               "note": "Disjoint TRAIN-split pilot to seal IC1 thresholds. MDE_census uses eligible-gene N; "
                       "SD_between from the pilot subsample. Linear f = G - mean(A,C,T)."}
    out.write_text(json.dumps(payload, indent=1, default=float))
    ckpt = Path(args.model); ckpt = ckpt if ckpt.is_absolute() else REPO / ckpt
    man = ExperimentManifest.create("splicebert_coupling:ic1_pilot", sys.argv, status="PILOT",
        config={"n_pilot": args.n_pilot, "n_seed": args.n_seed, "acc_ups": ACC_UPS},
        seeds={"base": 0}, notes=["IC1 pre-freeze pilot: shuffler unit tests + MDE/dose/seed/OOD."])
    man.add_input(str(ckpt / "pytorch_model.bin"))
    man.write(out.parent / "ic1_pilot.manifest.json")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
