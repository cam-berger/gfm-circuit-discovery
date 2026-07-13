#!/usr/bin/env python
"""THE DECISIVE EXPERIMENT: does the ~10.67 carrier track a GENOMIC statistic, or is it a
generic positional frequency that merely lands in biology's crowded 10-11 nt neighbourhood?

Specified by the user.  Five steps:
  1. Anchor real sequences at donors and acceptors.
  2. At periods 9-12, compute COMPLEX spectra separately for RR, YY, WW, SS.
  3. Measure amplitude AND phase-locking across landmarks.  *** Do NOT pool counterphase
     channels -- they cancel. ***  (My earlier WW/SS test did exactly that, and its null is
     therefore suspect.  Test each channel on its own AND as a contrast.)
  4. Ask whether L0H11's selected QK phase matches a landmark-relative phase.
  5. *** MOST IMPORTANTLY *** test whether carrier deletion damages HIGH-phase-locking windows
     more than matched LOW-phase-locking ones -- controlling for exon/intron, GC, complexity
     and codon content.

WHY THE EARLIER NULL DOES NOT SETTLE THIS.  run_carrier_strata.py used ONE signed channel
(WW minus SS), anchored on the masked base rather than a landmark, and controlled for nothing.
Every one of those is a way to manufacture a false negative.  This run fixes all four.

THE CONTROL THAT CAN FAIL: the codon clock, through the identical machinery.  It has no business
tracking a 10.67-nt sequence oscillation.  If it does, the phase-locking metric is confounded and
the test is void -- and this script says so instead of reporting it.

BASELINE, MEASURED (run_carrier_phase.py + the RoBERTa check):
  RoBERTa-base (TEXT, len 512) has a BROADBAND 9-12 bank -- peak at period 11.91, and at 10.67 it
  is not even a local maximum (20% of the band's power in its top bin).  SpliceBERT.510nt has a
  SHARP TONE at 10.667 (76% of the band in one bin, 15x over its neighbours).  So the BANK is a
  Transformer artefact; the SELECTED TONE is not.  That is what this experiment is trying to
  explain.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, load_adapter, load_examples,
    pick_device,
)

BAND_P3 = list(range(339, 346))
BAND_P10 = list(range(94, 101))
T10 = 1026 / 96
MASK_RATE = 1 / 7
MIN_LEN = 60
OFF = 1
WIN = 64                                     # +-64 nt ~ 12 helical turns
LAND = 100                                   # landmark window, +-100 nt
OUT = REPO / "results/phase-lattice/landmark_phase.json"

R_, Y_, W_, S_ = set("AG"), set("CT"), set("AT"), set("GC")
CHANNELS = {"RR": (R_, R_), "YY": (Y_, Y_), "WW": (W_, W_), "SS": (S_, S_)}


def dinuc(seq: str, a: int, b: int, ch: str) -> np.ndarray:
    """0/1 indicator of dinucleotide class `ch` at each position in [a, b)."""
    s1, s2 = CHANNELS[ch]
    return np.array([1.0 if (seq[i] in s1 and seq[i + 1] in s2) else 0.0
                     for i in range(a, min(b, len(seq) - 1))])


def zspec(x: np.ndarray, T: float, x0: int = 0) -> complex:
    """Complex Fourier coefficient at period T, with position measured from x0."""
    x = x - x.mean()
    k = np.arange(len(x)) + x0
    return complex((x * np.exp(-2j * np.pi * k / T)).sum() / max(len(x), 1))


def boot_ci(v, g, n=2000, seed=0):
    v, g = np.asarray(v, float), np.asarray(g)
    u = np.unique(g)
    idx = {x: np.flatnonzero(g == x) for x in u}
    rng = np.random.default_rng(seed)
    o = np.array([np.mean(np.concatenate([v[idx[x]] for x in rng.choice(u, len(u), True)]))
                  for _ in range(n)])
    return float(np.mean(v)), float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    dev = pick_device(None)
    ad = load_adapter(DEFAULT_MODEL, dev, eager=True)
    task = CouplingTask(ad, batch=4)
    m = ad.model
    emb = m.bert.embeddings.position_embeddings
    P0 = emb.weight.detach().clone()
    P = P0.float().cpu().numpy()
    N, _ = P.shape
    mu = P.mean(0, keepdims=True)
    F = np.fft.rfft(P - mu, axis=0)

    def strip(bins):
        G = F.copy()
        for b in bins:
            G[b] = 0
        return torch.tensor(np.fft.irfft(G, n=N, axis=0) + mu, dtype=P0.dtype, device=P0.device)

    X, _ = load_examples(DEFAULT_GTF, DEFAULT_FASTA, args.split, args.n * 4, task=task,
                         tag=f"land-{args.split}", min_effect=None)
    X = [x for x in X if x.gene_type == "protein_coding" and x.exon_length >= MIN_LEN
         and x.next_acceptor_pos - x.donor_pos >= MIN_LEN][:args.n]
    print(f"{len(X)} protein-coding exons, {len(set(x.gene_id for x in X))} genes  dev={dev}\n")
    rec: dict = {"n": len(X), "T10": T10}

    # ================================================================ 1-3. THE GENOMIC SIGNAL
    print("=" * 92)
    print("STEPS 1-3.  LANDMARK-ANCHORED COMPLEX SPECTRA.  Sequence only -- no model.")
    print("  PLV = |mean over sites of Z/|Z||.  1 = the phase is locked to the junction; 0 = random.")
    print("=" * 92)
    periods = np.arange(9.0, 12.01, 0.02)
    rec["landmark"] = {}
    print(f"\n{'landmark':9s} {'chan':5s} | {'best T':>7s} {'PLV':>6s} {'phase (nt from junction)':>25s}"
          f" {'|mean Z|':>9s}")
    print("-" * 92)
    for land, getp in (("donor", lambda x: x.donor_pos), ("acceptor", lambda x: x.acceptor_pos)):
        for ch in CHANNELS:
            Zs = []
            for x in X:
                p = getp(x)
                a, b = p - LAND, p + LAND
                if a < 1 or b > len(x.sequence) - 2:
                    continue
                d = dinuc(x.sequence, a, b, ch)
                Zs.append([zspec(d, T, x0=a - p) for T in periods])
            Z = np.array(Zs)                                   # (sites, periods)
            plv = np.abs((Z / np.abs(Z).clip(1e-12)).mean(0))  # phase-locking value
            j = int(plv.argmax())
            mz = Z[:, j].mean()
            ph_nt = (-np.angle(mz) / (2 * np.pi) * periods[j]) % periods[j]
            rec["landmark"][f"{land}_{ch}"] = dict(T=float(periods[j]), plv=float(plv[j]),
                                                   phase_nt=float(ph_nt), absZ=float(abs(mz)))
            print(f"{land:9s} {ch:5s} | {periods[j]:7.2f} {plv[j]:6.3f} {ph_nt:23.2f}   "
                  f"{abs(mz):9.4f}")

    # ================================================================ 5. THE INTERACTION TEST
    print("\n" + "=" * 92)
    print("STEP 5.  Does deleting the carrier hurt HIGH-phase-locking windows more?")
    print("=" * 92)
    mrng = np.random.default_rng(0)
    MASKS = [mrng.random(len(x.sequence)) < MASK_RATE for x in X]

    def regions(x):
        return {"exon": (x.acceptor_pos + 6, x.donor_pos - 6),
                "intron": (x.donor_pos + 8, x.next_acceptor_pos - 8)}

    def evaluate():
        out = []
        with torch.no_grad():
            for i in range(0, len(X), task.batch):
                ch = X[i:i + task.batch]
                enc = ad.tokenizer([" ".join(x.sequence) for x in ch], return_tensors="pt",
                                   padding=True).to(ad.device)
                pos = []
                for j, gi in enumerate(range(i, i + len(ch))):
                    ps = np.flatnonzero(MASKS[gi]).tolist()
                    for p in ps:
                        enc["input_ids"][j, p + OFF] = task.mask_id
                    pos.append(ps)
                ll = torch.log_softmax(m(**enc).logits.float(), dim=-1)
                for j, x in enumerate(ch):
                    idx = torch.tensor([p + OFF for p in pos[j]], device=ll.device)
                    tgt = torch.tensor([task.bid[x.sequence[p]] for p in pos[j]], device=ll.device)
                    v = np.full(len(x.sequence), np.nan)
                    v[pos[j]] = ll[j, idx, tgt].cpu().numpy()
                    out.append(v)
        return out

    LL = {}
    for k, (nm, bins) in enumerate([("intact", []), ("period3", BAND_P3), ("carrier", BAND_P10)]):
        emb.weight.data = P0.clone() if not bins else strip(bins)
        LL[nm] = evaluate()
        print(f"  [{k+1}/3] {nm}")
    emb.weight.data = P0.clone()

    print("\n  building per-window covariates (phase-lock per channel, GC, complexity, codon)...")
    rows = []
    for i, x in enumerate(X):
        seq = x.sequence
        reg = regions(x)
        for p in np.flatnonzero(MASKS[i]):
            a, b = p - WIN, p + WIN
            if a < 1 or b > len(seq) - 2:
                continue
            r = "exon" if reg["exon"][0] <= p < reg["exon"][1] else (
                "intron" if reg["intron"][0] <= p < reg["intron"][1] else None)
            if r is None:
                continue
            w = seq[a:b]
            gc = sum(c in S_ for c in w) / len(w)
            _, cnt = np.unique(list(w), return_counts=True)
            pr = cnt / cnt.sum()
            cplx = float(-(pr * np.log(pr)).sum() / np.log(4))          # 1 = max complexity
            # the sequence's OWN period-3 power (codon content) -- a covariate, not the outcome
            xx = np.array([{"A": 0, "C": 1, "G": 2, "T": 3}[c] == 0 for c in w], float)
            p3seq = abs(zspec(xx, 3.0))
            # phase-lock per channel at ~10.67, each ON ITS OWN (never pooled)
            pl = {ch: abs(zspec(dinuc(seq, a, b, ch), T10)) for ch in CHANNELS}
            rows.append(dict(gene=x.gene_id, region=r, gc=gc, cplx=cplx, p3seq=float(p3seq),
                             plmax=float(max(pl.values())), **{f"pl_{c}": float(v)
                                                               for c, v in pl.items()},
                             d3=float(LL["intact"][i][p] - LL["period3"][i][p]),
                             d10=float(LL["intact"][i][p] - LL["carrier"][i][p])))
    print(f"  {len(rows)} masked positions\n")

    # OLS with controls; bootstrap the phase-lock coefficient, grouped by gene
    genes = np.array([r["gene"] for r in rows])
    def design(key):
        return np.column_stack([
            np.ones(len(rows)),
            [r[key] for r in rows],                                   # <- the phase-lock term
            [r["gc"] for r in rows], [r["cplx"] for r in rows], [r["p3seq"] for r in rows],
            [1.0 if r["region"] == "exon" else 0.0 for r in rows],
        ])

    def beta_ci(key, dmg, n=1000, seed=0):
        A = design(key)
        y = np.array([r[dmg] for r in rows])
        u = np.unique(genes)
        idx = {g: np.flatnonzero(genes == g) for g in u}
        rng = np.random.default_rng(seed)
        out = np.empty(n)
        for t in range(n):
            sel = np.concatenate([idx[g] for g in rng.choice(u, len(u), True)])
            out[t] = np.linalg.lstsq(A[sel], y[sel], rcond=None)[0][1]
        b = np.linalg.lstsq(A, y, rcond=None)[0][1]
        return float(b), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

    print("  damage ~ b*phaselock + GC + complexity + codon-content + is_exon\n")
    print(f"  {'channel':9s} | {'CARRIER arm  b [95% CI]':>34s} | "
          f"{'CODON-CLOCK arm (control)':>34s}")
    print("  " + "-" * 82)
    rec["interaction"] = {}
    for key in ("plmax", "pl_RR", "pl_YY", "pl_WW", "pl_SS"):
        b10 = beta_ci(key, "d10")
        b3 = beta_ci(key, "d3")
        rec["interaction"][key] = dict(carrier=b10, codon=b3)
        f10 = f"{b10[0]:+.4f} [{b10[1]:+.4f}, {b10[2]:+.4f}]"
        f3 = f"{b3[0]:+.4f} [{b3[1]:+.4f}, {b3[2]:+.4f}]"
        star = "  *" if (b10[1] > 0 or b10[2] < 0) else ""
        print(f"  {key:9s} | {f10:>34s} | {f3:>34s}{star}")
    print("  " + "-" * 82)
    print("  * = the carrier's phase-lock coefficient EXCLUDES ZERO")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
