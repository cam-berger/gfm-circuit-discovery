#!/usr/bin/env python
"""WHEN WAS THE PERIOD-3 CARRIER LEARNED, AND WHAT SUBSPACE IS IT IN?

ESTABLISHED (results/phase-lattice/):
  * SpliceBERT's position table P [1026 x 512] carries a PERIOD-3 CARRIER: Fourier bins 339-345,
    847x median bin power, the ONLY high-frequency structure in P's top-10 modes.
  * Six heads read it out.  Deleting the band collapses their comb amplitude to 3.4%; deleting
    an energy-matched band leaves it at 99.8%.
  * Deleting it costs 0.159 nats in CODING exons vs 0.042 in introns (energy-matched control:
    0.029/0.031, i.e. EQUAL) and destroys the donor-acceptor coupling (0.946 -> 0.027), which is
    rescued by restoring L2H15 -- the ADDRESSER -- alone (0.363 vs 0.007 for random heads).

THREE QUESTIONS LEFT, ALL ANSWERABLE FROM WEIGHTS.

Q1. WHEN WAS IT LEARNED?  SpliceBERT was trained in two stages: multi-species pre-mRNA at 510nt,
    then extended to variable length up to 1024nt.  A human-only 510nt model was also released.
    All three checkpoints are on disk.  So:

        1024nt (multispecies + long-context)   <- where we found the carrier
        510nt  (multispecies, stage 1)
        510nt  (HUMAN ONLY)
        random init                            <- the floor

    NOTE: 512 rows is NOT divisible by 3, so there is no exact FFT bin in the 510nt checkpoints.
    Project directly onto the period-3 basis instead -- exact at any N:

        C(f) = sum_i Ptilde_i cos(2 pi f i),   S(f) = sum_i Ptilde_i sin(2 pi f i)
        power(f) = (||C||^2 + ||S||^2) / N

    and build the null by sweeping f across the whole band.  Enrichment = power(1/3) / median.

Q2. WHAT SUBSPACE IS IT IN?  SVD the reconstructed carrier.  If it is low-rank, the "clock" is a
    handful of directions in the 512-dim residual stream, and every head that reads phase must
    read THOSE directions.

Q3. WHICH HEADS READ IT -- COMPUTED THE WAY THE MODEL ACTUALLY SEES IT?
    The earlier R(k) projected the RAW P through W_Q/W_K.  The model never sees raw P.  It sees

        z_i = LayerNorm( E_base(x_i) + P_i + E_segment )

    LayerNorm's rescaling is content-dependent, so the raw-P calculation is an approximation.
    Fixed here two ways:
      * layer 0: average the POST-LayerNorm embedding over A/C/G/T at each position, then project.
      * layers 1-2: those heads do not see P either -- they see the residual stream.  So use the
        EMPIRICAL position-conditional mean hidden state (averaged over many random sequences)
        at each layer's input.  That is what those heads actually read.
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

from transformers import AutoModelForMaskedLM, AutoTokenizer, BertConfig  # noqa: E402
from transformers import BertForMaskedLM  # noqa: E402

BASE = REPO / "data/models/splicebert_official_archive/extracted/models"
CKPTS = {
    "SpliceBERT.1024nt": BASE / "SpliceBERT.1024nt",
    "SpliceBERT.510nt (multispecies)": BASE / "SpliceBERT.510nt",
    "SpliceBERT-human.510nt": BASE / "SpliceBERT-human.510nt",
}
COMBS = [(0, 0), (0, 11), (1, 10), (1, 12), (2, 2), (2, 15)]
BAND = list(range(339, 346))


def period_power(Pt: np.ndarray, f: float) -> float:
    """Direct projection onto the period-1/f basis.  Exact at any N (no integer bin needed)."""
    N = Pt.shape[0]
    i = np.arange(N)
    C = np.cos(2 * np.pi * f * i) @ Pt
    S = np.sin(2 * np.pi * f * i) @ Pt
    return float((C @ C + S @ S) / N)


def carrier_enrichment(P: np.ndarray) -> tuple[float, float, int]:
    """power(1/3) / median power over a frequency sweep, and the rank of 1/3 in that sweep."""
    Pt = P - P.mean(0, keepdims=True)
    N = Pt.shape[0]
    fs = np.arange(1, N // 2) / N                       # every resolvable frequency
    pw = np.array([period_power(Pt, f) for f in fs])
    p3 = period_power(Pt, 1 / 3)
    med = float(np.median(pw))
    rank = int(np.sum(pw > p3)) + 1
    return p3 / med, p3, rank


def mod3_separation(P: np.ndarray) -> float:
    """Exact, length-free: between-group / within-group variance of the mod-3 position groups.

    A carrier makes positions of the same residue class mod 3 more similar to each other than to
    the other classes.  This needs no Fourier bin and no integer divisibility.
    """
    Pt = P - P.mean(0, keepdims=True)
    N = Pt.shape[0]
    idx = np.arange(N)
    mus = np.stack([Pt[idx % 3 == r].mean(0) for r in range(3)])
    between = float(np.mean(np.sum((mus - mus.mean(0)) ** 2, axis=1)))
    within = float(np.mean([np.mean(np.sum((Pt[idx % 3 == r] - mus[r]) ** 2, axis=1))
                            for r in range(3)]))
    return between / max(within, 1e-9)


def mod_k_null(P: np.ndarray, ks=(2, 4, 5, 7, 8, 10, 11)) -> dict[int, float]:
    """The same statistic for OTHER moduli.  If mod-3 is not special, these match it."""
    Pt = P - P.mean(0, keepdims=True)
    N = Pt.shape[0]
    idx = np.arange(N)
    out = {}
    for k in ks:
        mus = np.stack([Pt[idx % k == r].mean(0) for r in range(k)])
        between = float(np.mean(np.sum((mus - mus.mean(0)) ** 2, axis=1)))
        within = float(np.mean([np.mean(np.sum((Pt[idx % k == r] - mus[r]) ** 2, axis=1))
                                for r in range(k)]))
        out[k] = between / max(within, 1e-9)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, default=16, help="random seqs for the empirical readout")
    ap.add_argument("--out", default="results/phase-lattice/carrier_provenance.json")
    args = ap.parse_args()

    R: dict = {}

    # ============================================================ Q1: WHEN WAS IT LEARNED?
    print("=" * 96)
    print("Q1.  WHEN WAS THE CARRIER LEARNED?")
    print("     period-3 power / median power over a full frequency sweep (exact at any N --")
    print("     512 rows has no integer period-3 bin, so we project onto cos/sin directly).")
    print(f"\n  {'checkpoint':34s} {'rows':>5s} {'nt rows':>8s} {'enrich':>8s} {'rank':>6s} "
          f"{'mod-3 sep':>10s}")
    prov: dict = {}
    for name, path in CKPTS.items():
        m = AutoModelForMaskedLM.from_pretrained(str(path))
        P = m.bert.embeddings.position_embeddings.weight.detach().float().numpy()
        N = P.shape[0]
        nt = P[1:N - 1]                                  # drop [CLS] row 0 and [SEP] last row
        en, p3, rank = carrier_enrichment(nt)
        sep = mod3_separation(nt)
        nulls = mod_k_null(nt)
        prov[name] = dict(rows=N, nt_rows=nt.shape[0], enrichment=en, power=p3, rank=rank,
                          mod3_sep=sep, mod_k_null=nulls)
        print(f"  {name:34s} {N:5d} {nt.shape[0]:8d} {en:8.1f}x {rank:6d} {sep:10.3f}")
        del m

    cfg = BertConfig.from_pretrained(str(CKPTS["SpliceBERT.1024nt"]))
    torch.manual_seed(0)
    rm = BertForMaskedLM(cfg)                            # RANDOM INIT -- the floor
    Pr = rm.bert.embeddings.position_embeddings.weight.detach().float().numpy()
    ntr = Pr[1:Pr.shape[0] - 1]
    en, p3, rank = carrier_enrichment(ntr)
    sep = mod3_separation(ntr)
    prov["random init (1024nt config)"] = dict(rows=Pr.shape[0], nt_rows=ntr.shape[0],
                                               enrichment=en, power=p3, rank=rank, mod3_sep=sep,
                                               mod_k_null=mod_k_null(ntr))
    print(f"  {'random init (1024nt config)':34s} {Pr.shape[0]:5d} {ntr.shape[0]:8d} "
          f"{en:8.1f}x {rank:6d} {sep:10.3f}")

    print("\n  mod-3 separation vs OTHER moduli (the null -- if 3 is not special they match):")
    print(f"    {'checkpoint':34s} " + " ".join(f"{'k='+str(k):>7s}" for k in (2, 3, 4, 5, 7)))
    for name in prov:
        v = prov[name]
        row = []
        for k in (2, 3, 4, 5, 7):
            x = v["mod3_sep"] if k == 3 else v["mod_k_null"][k]
            row.append(f"{x:7.3f}")
        star = "  <== " if v["mod3_sep"] > 5 * max(v["mod_k_null"][k] for k in (2, 4, 5, 7)) else ""
        print(f"    {name:34s} " + " ".join(row) + star)
    R["provenance"] = prov

    # ============================================================ Q2: THE SUBSPACE
    print("\n" + "=" * 96)
    print("Q2.  WHAT SUBSPACE IS THE CLOCK IN?   SVD of the reconstructed carrier.")
    m = AutoModelForMaskedLM.from_pretrained(str(CKPTS["SpliceBERT.1024nt"]),
                                             attn_implementation="eager").eval()
    P = m.bert.embeddings.position_embeddings.weight.detach().float().numpy()
    N = P.shape[0]
    mu = P.mean(0, keepdims=True)
    F = np.fft.rfft(P - mu, axis=0)
    G = np.zeros_like(F)
    G[BAND] = F[BAND]
    CAR = np.fft.irfft(G, n=N, axis=0)                   # the carrier alone, [1026, 512]
    U, S, Vh = np.linalg.svd(CAR, full_matrices=False)
    ev = S ** 2 / (S ** 2).sum()
    print(f"  carrier energy by singular value: " + "  ".join(f"{100*e:.1f}%" for e in ev[:8]))
    k90 = int(np.searchsorted(np.cumsum(ev), 0.90) + 1)
    print(f"  effective rank (90% of carrier energy): **{k90}** of 512 embedding dimensions")
    print(f"  => the clock lives in a {k90}-dimensional subspace of the residual stream.")
    R["svd"] = dict(explained=[float(x) for x in ev[:12]], rank90=k90)

    # ============================================================ Q3: WHO READS IT (LayerNorm-correct)
    print("\n" + "=" * 96)
    print("Q3.  WHICH HEADS READ THE CARRIER -- as the model actually sees it?")
    print("     The model never sees raw P. It sees LayerNorm(E_base(x) + P_i + E_seg), and")
    print("     layer-1/2 heads see the RESIDUAL STREAM, not P at all. So:")
    print("       layer 0   : average the POST-LayerNorm embedding over A/C/G/T at each position")
    print("       layers 1-2: the EMPIRICAL position-conditional mean hidden state, from real")
    print("                   forward passes over random sequences")

    tok = AutoTokenizer.from_pretrained(str(CKPTS["SpliceBERT.1024nt"]))
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACGT"), 1024)) for _ in range(args.seqs)]
    enc = tok([" ".join(s) for s in seqs], return_tensors="pt", padding=True)
    with torch.no_grad():
        hs = m.bert(**enc, output_hidden_states=True).hidden_states   # tuple[L+1] [B,S,512]
    # position-conditional mean representation at the INPUT of each layer, averaged over content
    Hbar = [h.mean(0).float().numpy() for h in hs]                    # [L+1][S,512]

    nH, dh = m.config.num_attention_heads, m.config.hidden_size // m.config.num_attention_heads

    def Rk(l, h, Hin, span=12, lo=200, hi=800):
        sa = m.bert.encoder.layer[l].attention.self
        Wq = sa.query.weight.detach().float().numpy()[h * dh:(h + 1) * dh]
        bq = sa.query.bias.detach().float().numpy()[h * dh:(h + 1) * dh]
        Wk = sa.key.weight.detach().float().numpy()[h * dh:(h + 1) * dh]
        bk = sa.key.bias.detach().float().numpy()[h * dh:(h + 1) * dh]
        Q = Hin @ Wq.T + bq
        K = Hin @ Wk.T + bk
        idx = np.arange(lo, hi)
        return np.array([float(np.mean(np.sum(Q[idx] * K[idx + k], axis=1))) / np.sqrt(dh)
                         for k in range(-span, span + 1)])

    def amp(r, span=12):
        ks = np.arange(-span, span + 1)
        mm = ks != 0
        g = [r[mm][(ks[mm] % 3) == c].mean() for c in range(3)]
        return float(max(g) - min(g))

    print(f"\n  {'head':7s} {'comb amp (raw P)':>17s} {'comb amp (as seen)':>19s}   "
          f"R(k) k=-6..+6, as the model sees it")
    heads_out = []
    for (l, h) in COMBS:
        r_raw = Rk(l, h, P)                    # the old, approximate calculation
        r_true = Rk(l, h, Hbar[l])             # what the head actually reads
        prof = " ".join(f"{r_true[12 + k]:+5.2f}" for k in range(-6, 7))
        print(f"  L{l}H{h:<4d} {amp(r_raw):17.3f} {amp(r_true):19.3f}   {prof}")
        heads_out.append(dict(layer=l, head=h, amp_rawP=amp(r_raw), amp_asseen=amp(r_true)))

    # how much of each head's QK does the CARRIER SUBSPACE explain?
    V = Vh[:k90]                                          # the carrier directions
    print(f"\n  fraction of each head's QK bilinear form that lives in the {k90}-dim carrier "
          f"subspace:")
    print(f"  (M = Wq^T Wk;  ||V M V^T||_F^2 / ||M||_F^2.  chance ~ ({k90}/512)^2 = "
          f"{(k90/512)**2:.4f})")
    scores = []
    for l in range(m.config.num_hidden_layers):
        for h in range(nH):
            sa = m.bert.encoder.layer[l].attention.self
            Wq = sa.query.weight.detach().float().numpy()[h * dh:(h + 1) * dh]
            Wk = sa.key.weight.detach().float().numpy()[h * dh:(h + 1) * dh]
            M = Wq.T @ Wk
            proj = V @ M @ V.T
            scores.append(((l, h), float((proj ** 2).sum() / (M ** 2).sum())))
    scores.sort(key=lambda t: -t[1])
    combset = set(COMBS)
    print(f"    {'rank':>4s} {'head':7s} {'carrier frac':>13s}   is it one of the 6 combs?")
    for r, ((l, h), s) in enumerate(scores[:12], 1):
        print(f"    {r:4d} L{l}H{h:<4d} {s:13.4f}   {'YES  <==' if (l,h) in combset else ''}")
    n_top = sum(1 for (lh, _) in scores[:12] if lh in combset)
    print(f"\n  {n_top}/6 combs are in the top 12 of 96 heads by carrier-subspace loading.")
    R["qk_carrier_loading"] = [dict(layer=l, head=h, frac=s, comb=(l, h) in combset)
                               for (l, h), s in scores]
    R["heads"] = heads_out

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=1))
    print("\n" + "=" * 96)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
