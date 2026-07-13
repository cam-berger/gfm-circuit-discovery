#!/usr/bin/env python
"""WHAT IS THE PERIOD-10.69 CARRIER FOR?  Four hypotheses, and the ones this run can kill.

The user's framing:
    period 3   may be a GENETIC clock       (reading frame)
    period 10.65 may be a STRUCTURAL clock  (a rotation)

and the candidate rotations all sit near the measured period:
    protein alpha-helix   3.6 residues/turn -> 3.6 x 3 = 10.80 nt of coding sequence
    B-DNA (solution)      ~10.5 bp/turn
    A-form RNA            ~11   bp/turn
    a generic ruler       no biological meaning at all

WHAT THE SPECTRUM ALREADY SAYS (no compute needed)
--------------------------------------------------
P has N = 1026 rows, so every candidate period maps to an EXACT Fourier bin and a pure tone
there would land on it with NO leakage:
    10.80 (alpha-helix)      -> bin  95     ->    2.5x the median bin   <- DEAD
    11.00 (A-form RNA)       -> bins 93/94  ->    1.8x / 44.3x          <- DEAD
    10.20 (NUCLEOSOMAL DNA)  -> bins 100/101->   14.6x /  3.4x          <- DEAD
    10.50 (B-DNA, SOLUTION)  -> bins 97/98  ->  105.0x / 131.3x         <- LIVE
    10.69 (the model)        -> bin  96     ->  546.0x                  <- THE CARRIER

  (CORRECTED: an earlier version of this note quoted B-DNA at 10.2 -- the NUCLEOSOMAL repeat --
   and left the canonical solution value, 10.5, out of the table.  10.5 is NOT dead.)

But run_carrier_geometry.py then shows bins 97/98 are the bin-96 carrier's SIDEBAND, not a
separate 10.5 tone: they occupy the SAME embedding directions (subspace overlap 0.989, against a
0.990 calibration on the codon clock's known 341/343 sidebands).  So P holds exactly ONE tone
here, at 10.69 -- neither 10.5 nor 10.8.  Both are ~0.15 nt away.  Frequency cannot adjudicate.
Function has to.

THE TWO TESTS HERE
------------------
1. TRANSLATION TEST (kills or crowns the alpha-helix).
   An alpha-helical periodicity exists ONLY in translated, in-frame coding sequence.  It is
   meaningless in an intron.  So:
       alpha-helix  => deleting the carrier hurts CODING >> INTRON
       DNA / RNA / generic ruler => it hurts them EQUALLY
   The POSITIVE CONTROL is the codon clock itself, through the identical machinery: it must show
   the coding-specific damage the alpha-helix hypothesis predicts.  If the assay can detect
   coding-specificity for period 3 and detects none for period 10.69, the negative is POWERED --
   and that is the whole point.  A null without a positive control is not a result.

2. HELICAL-PHASING TEST (the chromatin reading), and it needs NO external data.
   Nucleosome-positioning sequence carries its own ~10.5-bp WW/SS dinucleotide periodicity
   (W = A/T, S = G/C): AA/TT/AT dinucleotides recur about once per helical turn, which is what
   lets DNA bend around the histone octamer.  We can measure that IN THE SEQUENCE ITSELF.
       chromatin/DNA-helix => the carrier is MORE useful where the sequence is helically phased,
                              so deleting it should hurt MORE in high-phasing windows
       generic ruler       => no relationship
   The control that CAN fail: the SAME stratification is applied to the codon-clock arm.  The
   codon clock has no business tracking WW/SS phasing.  If it does, the phasing metric is
   confounded and the test is VOID -- and this script says so rather than reporting it.
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
MASK_RATE = 1 / 7
MIN_LEN = 60
OFF = 1                                  # [CLS]
CHANCE = float(np.log(0.25))
T10 = 1026 / 96                          # 10.6875
PHASE_WIN = 48                           # +-48 nt ~ 9 helical turns
OUT = REPO / "results/phase-lattice/carrier_strata.json"

W_SET, S_SET = set("AT"), set("GC")


def ww_ss_phasing(seq: str, centre: int) -> float:
    """Strength of the sequence's OWN ~10.69-nt WW/SS dinucleotide periodicity around `centre`.

    s(i) = +1 if bases (i, i+1) are both W (A/T); -1 if both S (G/C); 0 otherwise.
    Return |projection of s onto e^{2*pi*i*k/T10}| / n  -- the classic nucleosome/helical-phasing
    signal, computed from the sequence and nothing else.
    """
    a, b = max(0, centre - PHASE_WIN), min(len(seq) - 1, centre + PHASE_WIN)
    if b - a < 40:
        return float("nan")
    s = np.zeros(b - a)
    for i in range(a, b):
        p, q = seq[i], seq[i + 1]
        if p in W_SET and q in W_SET:
            s[i - a] = 1.0
        elif p in S_SET and q in S_SET:
            s[i - a] = -1.0
    if not s.any():
        return 0.0
    k = np.arange(len(s))
    z = (s * np.exp(-2j * np.pi * k / T10)).sum() / len(s)
    return float(abs(z))


def boot_ci(vals, genes, n=2000, seed=0):
    """Gene-grouped bootstrap.  Exons of one gene are not independent."""
    vals, genes = np.asarray(vals, float), np.asarray(genes)
    uniq = np.unique(genes)
    idx = {g: np.flatnonzero(genes == g) for g in uniq}
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        out[i] = np.mean(np.concatenate([vals[idx[g]] for g in pick]))
    return float(np.mean(vals)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
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
                         tag=f"strata-{args.split}", min_effect=None)
    X = [x for x in X if x.gene_type == "protein_coding" and x.exon_length >= MIN_LEN
         and x.next_acceptor_pos - x.donor_pos >= MIN_LEN][:args.n]
    print(f"{len(X)} protein-coding exons, {len(set(x.gene_id for x in X))} genes  device={dev}\n")

    mrng = np.random.default_rng(args.seed)
    MASKS = [mrng.random(len(x.sequence)) < MASK_RATE for x in X]

    def regions(x):
        return {"exon": (x.acceptor_pos + 6, x.donor_pos - 6),
                "intron": (x.donor_pos + 8, x.next_acceptor_pos - 8)}

    def evaluate() -> list[np.ndarray]:
        """per-EXAMPLE, per-POSITION log-prob of the true base (NaN where not masked)."""
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

    ARMS = [("intact", []), ("period3", BAND_P3), ("carrier10", BAND_P10)]
    LL: dict[str, list[np.ndarray]] = {}
    for k, (name, bins) in enumerate(ARMS):
        emb.weight.data = P0.clone() if not bins else strip(bins)
        LL[name] = evaluate()
        print(f"  [{k+1}/3] {name}")
    emb.weight.data = P0.clone()

    # ---------------------------------------------------------------- the sequence's own phasing
    print("\ncomputing WW/SS helical phasing from the sequence (no external data)...")
    PH = [np.array([ww_ss_phasing(x.sequence, p) for p in np.flatnonzero(MASKS[i])])
          for i, x in enumerate(X)]

    # ---------------------------------------------------------------- gather
    rec: dict = {"n": len(X), "split": args.split, "T10": T10,
                 "band_p3": BAND_P3, "band_p10": BAND_P10}

    # --- TEST 1: the translation test (exon vs intron), with a gene-grouped bootstrap
    print("\n" + "=" * 78)
    print("TEST 1 — THE TRANSLATION TEST.  Does the carrier serve TRANSLATED sequence?")
    print("=" * 78)
    print(f"\n{'arm':11s} {'region':7s} {'damage (nats)':>22s}   {'DiD  exon - intron':>26s}")
    print("-" * 78)
    rec["translation"] = {}
    for arm in ("period3", "carrier10"):
        per_region, per_gene = {}, {}
        for r in ("exon", "intron"):
            v, g = [], []
            for i, x in enumerate(X):
                a, b = regions(x)[r]
                if b - a < 24:
                    continue
                d = LL["intact"][i][a:b] - LL[arm][i][a:b]
                d = d[~np.isnan(d)]
                if len(d):
                    v.append(float(d.mean()))
                    g.append(x.gene_id)
            per_region[r], per_gene[r] = v, g
        # DiD per gene, so the CI is on the CONTRAST, not on two independent means
        genes = sorted(set(per_gene["exon"]) & set(per_gene["intron"]))
        ge = {g: [] for g in genes}
        gi = {g: [] for g in genes}
        for v, g in zip(per_region["exon"], per_gene["exon"]):
            if g in ge:
                ge[g].append(v)
        for v, g in zip(per_region["intron"], per_gene["intron"]):
            if g in gi:
                gi[g].append(v)
        did = [np.mean(ge[g]) - np.mean(gi[g]) for g in genes]
        out = {}
        for r in ("exon", "intron"):
            mn, lo, hi = boot_ci(per_region[r], per_gene[r])
            out[r] = dict(damage=mn, lo=lo, hi=hi, n=len(per_region[r]))
        dm, dlo, dhi = boot_ci(did, genes)
        out["did"] = dict(mean=dm, lo=dlo, hi=dhi, n_genes=len(genes))
        rec["translation"][arm] = out
        for r in ("exon", "intron"):
            o = out[r]
            tag = f"{o['damage']:+.4f} [{o['lo']:+.4f}, {o['hi']:+.4f}]"
            extra = f"{dm:+.4f} [{dlo:+.4f}, {dhi:+.4f}]" if r == "exon" else ""
            print(f"{arm:11s} {r:7s} {tag:>22s}   {extra:>26s}")
        print("-" * 78)

    # --- TEST 2: helical phasing
    print("\n" + "=" * 78)
    print("TEST 2 — THE HELICAL-PHASING TEST.  Is the carrier worth more where the SEQUENCE")
    print("         itself is helically phased (the nucleosome/WW-SS signal)?")
    print("=" * 78)
    ph_all = np.concatenate([p[~np.isnan(p)] for p in PH])
    q1, q2 = np.percentile(ph_all, [33.3, 66.7])
    print(f"\nWW/SS phasing at period {T10:.2f}: tertile cuts {q1:.4f} / {q2:.4f}\n")
    print(f"{'arm':11s} {'phasing':10s} {'damage (nats)':>24s}   {'high - low':>24s}")
    print("-" * 78)
    rec["phasing"] = {"cuts": [float(q1), float(q2)]}
    for arm in ("period3", "carrier10"):
        buckets: dict = {"low": ([], []), "mid": ([], []), "high": ([], [])}
        gdiff: dict = {}
        for i, x in enumerate(X):
            ps = np.flatnonzero(MASKS[i])
            d = LL["intact"][i][ps] - LL[arm][i][ps]
            ph = PH[i]
            keep = ~np.isnan(d) & ~np.isnan(ph)
            for name, sel in (("low", ph <= q1), ("mid", (ph > q1) & (ph <= q2)),
                              ("high", ph > q2)):
                s = sel & keep
                if s.sum() >= 3:
                    buckets[name][0].append(float(d[s].mean()))
                    buckets[name][1].append(x.gene_id)
            hi_s, lo_s = (ph > q2) & keep, (ph <= q1) & keep
            if hi_s.sum() >= 3 and lo_s.sum() >= 3:
                gdiff.setdefault(x.gene_id, []).append(
                    float(d[hi_s].mean() - d[lo_s].mean()))
        out = {}
        for name in ("low", "mid", "high"):
            mn, lo, hi = boot_ci(*buckets[name])
            out[name] = dict(damage=mn, lo=lo, hi=hi)
        gk = sorted(gdiff)
        dm, dlo, dhi = boot_ci([np.mean(gdiff[g]) for g in gk], gk)
        out["high_minus_low"] = dict(mean=dm, lo=dlo, hi=dhi, n_genes=len(gk))
        rec["phasing"][arm] = out
        for name in ("low", "mid", "high"):
            o = out[name]
            tag = f"{o['damage']:+.4f} [{o['lo']:+.4f}, {o['hi']:+.4f}]"
            extra = f"{dm:+.4f} [{dlo:+.4f}, {dhi:+.4f}]" if name == "low" else ""
            print(f"{arm:11s} {name:10s} {tag:>24s}   {extra:>24s}")
        print("-" * 78)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1))
    print(f"\nsaved -> {OUT}")

    # ---------------------------------------------------------------- the verdict
    p3, c10 = rec["translation"]["period3"], rec["translation"]["carrier10"]
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    powered = p3["did"]["lo"] > 0.02
    print(f"\n  positive control (codon clock): DiD {p3['did']['mean']:+.4f} "
          f"[{p3['did']['lo']:+.4f}, {p3['did']['hi']:+.4f}]")
    if not powered:
        print("  ** THE ASSAY DID NOT DETECT CODING-SPECIFICITY FOR THE CODON CLOCK. **")
        print("  ** It has no power.  The carrier's null below means NOTHING.  VOID.  **")
        return
    print("  -> the assay CAN see coding-specificity.  The carrier's null is therefore powered.\n")
    print(f"  the carrier (period 10.69):     DiD {c10['did']['mean']:+.4f} "
          f"[{c10['did']['lo']:+.4f}, {c10['did']['hi']:+.4f}]")
    if c10["did"]["lo"] <= 0 <= c10["did"]["hi"]:
        print("  -> CI includes zero.  The carrier does NOT preferentially serve translated")
        print("     sequence.  **The protein alpha-helix hypothesis is not supported.**")
    else:
        print("  -> CI EXCLUDES zero.  The carrier IS coding-specific -- the alpha-helix")
        print("     hypothesis gains substantially.  Follow up with secondary-structure strata.")

    ph3, ph10 = rec["phasing"]["period3"], rec["phasing"]["carrier10"]
    print(f"\n  helical-phasing control (codon clock): high-low "
          f"{ph3['high_minus_low']['mean']:+.4f} "
          f"[{ph3['high_minus_low']['lo']:+.4f}, {ph3['high_minus_low']['hi']:+.4f}]")
    conf = not (ph3["high_minus_low"]["lo"] <= 0 <= ph3["high_minus_low"]["hi"])
    if conf:
        print("  ** THE CODON CLOCK ALSO TRACKS WW/SS PHASING.  The metric is CONFOUNDED. **")
        print("  ** The phasing test below is VOID. **")
    else:
        print("  -> the codon clock does NOT track WW/SS phasing, as it should not.  Clean.\n")
        print(f"  the carrier (period 10.69):           high-low "
              f"{ph10['high_minus_low']['mean']:+.4f} "
              f"[{ph10['high_minus_low']['lo']:+.4f}, {ph10['high_minus_low']['hi']:+.4f}]")
        if ph10["high_minus_low"]["lo"] > 0:
            print("  -> the carrier is worth MORE in helically phased sequence.")
            print("     **Consistent with the DNA/chromatin reading.**")
        else:
            print("  -> no relationship.  **The chromatin reading is not supported either;**")
            print("     this looks like a generic positional ruler.")


if __name__ == "__main__":
    main()
