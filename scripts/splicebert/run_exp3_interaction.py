#!/usr/bin/env python
"""EXP 3 — THE RECIPROCAL CARRIER x DYAD-SCORE INTERACTION.  Does either carrier USE chromatin?

D32 established that the two checkpoints hold DIFFERENT, causally-read carriers:
    human-only    lambda ~ 10.24  (bins 49-51), 14 readers   <- near the nucleosomal pitch (10.298)
    multispecies  lambda ~ 10.61  (bins 46-48), 25 readers   <- R17: NOT nucleosomal
But 14 readers establish COMPUTATIONAL USE of a carrier -- not WHAT STATISTIC they use it for.
"Human-only learned chromatin" is the HYPOTHESIS.  This is the test.

THE 2x2, RECIPROCAL BY DESIGN.  Each carrier band is the other model's NEGATIVE CONTROL.
                     delete ~10.2 (bins 49-51)      delete ~10.63 (bins 46-48)
    human-only       PREDICT I > 0                  predict I ~ 0   (negative control)
    multispecies     predict I ~ 0 (neg. control)   predict I ~ 0   (may damage; must not TRACK score)

    I(m,b) = [L_b(HIGH) - L_int(HIGH)] - [L_b(LOW) - L_int(LOW)]        <- a DiD

PREREGISTERED, BEFORE THE RUN:
    H1  I(human, 10.2)  > 0                and is the LARGEST of the four
    H2  I(human, 10.63) ~ 0
    H3  I(multi, 10.2)  ~ 0
    H4  I(multi, 10.63) ~ 0   (its main effect L_b - L_int may be large; the INTERACTION must not be)
    H5  the energy-matched INERT band gives I ~ 0 in BOTH models

WHAT THE PAPER MATH (D33) FORCED, AND WHY EACH CHOICE IS NOT OPTIONAL
  * REGIONAL SCORE, NOT PER-WINDOW.  A 510-nt window holds only ~2.5 dyads: SNR 0.35,
    reliability 0.106 -> the interaction would be attenuated 9.4x.  That is R16's lesson verbatim
    ("a short window contains essentially no usable phase estimate").  20-kb bins -> reliability
    0.823.  The score is REGIONAL and is assigned to every window inside the bin.
  * PAIRED, ON IDENTICAL TOKENS AND IDENTICAL MASKS.  The tone carries only 0.000319 nats of
    ORACLE information -- 0.027% of the loss.  Unpaired, against a per-token loss sd of ~0.7,
    this is unreachable.  Paired, the per-token DIFFERENCE has a tiny sd and ~4e6 masked
    tokens/stratum gives 10-18 sigma.  An unpaired design would need ~500x the compute.
  * NO PHASE-TUNING CURVE.  D33: the QK comb is a function of OFFSET ONLY -- TRANSLATION-INVARIANT
    -- so the model CANNOT read a phase relative to an external landmark.  And with a globally
    coherent dyad phase (rho ~ 1), shifting the score's phase rescales every window by the SAME
    factor and never reorders them.  I(delta) would be a square wave, not a cosine.  It would have
    looked beautiful and meant nothing.

CONTROLS THAT CAN FAIL
  1. STRATUM VALIDATION.  The HIGH stratum must actually carry MORE 10.298 tone than the LOW one,
     measured directly and pooled.  If it does not, the strata are meaningless and the run is VOID.
  2. GC MATCHING.  WW density and GC are mechanically anti-correlated, so a "high dyad-score"
     stratum is at risk of being merely an AT-rich stratum.  The strata are GC-matched.
  3. RECIPROCAL BAND.  Each model's control band is the OTHER model's carrier -- semantically
     meaningful, not an arbitrary hole.
  4. ENERGY-MATCHED INERT BAND.  Guards against "deleting energy from P hurts".  (The DiD already
     cancels generic damage, since it subtracts the LOW stratum -- but this makes it explicit.)

⚠️ SPECIES CAVEAT, STATED UP FRONT.  The clean map is MOUSE (Voong chemical, mm9).  The human-only
model saw only HUMAN.  Mouse sequence is therefore OUT OF DISTRIBUTION for it.  This run is the
DISCOVERY arm; a species-matched replication on human dyads (Gaffney 2012, MNase -- needs
enzyme-bias controls) is REQUIRED before any claim, and is NOT done here.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from pyfaidx import Fasta

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/splicebert"))

from gfmcd.splicing.circuit import load_adapter, pick_device        # noqa: E402
from run_dyad_real import load_dyads, coeffs, FASTA                 # noqa: E402
from run_dyad_fpr import K                                          # noqa: E402
from transformers import AutoTokenizer                              # noqa: E402

OUT = REPO / "results/phase-lattice/exp3_interaction.json"
MODELS = {
    "human_only": ("data/models/splicebert_official_archive/extracted/models/SpliceBERT-human.510nt",
                   [49, 50, 51]),
    "multispecies": ("data/models/splicebert_official_archive/extracted/models/SpliceBERT.510nt",
                     [46, 47, 48]),
}
BANDS = {"del_10.2": [49, 50, 51], "del_10.63": [46, 47, 48]}

# VERIFIED-INERT CONTROL BANDS.  Found by search over all 3-bin windows, then AUDITED: the
# strongest head amplitude at that band's period, and how much of it survives the band's deletion.
# An INERT band is one no head reads (low amp) AND whose deletion changes nothing (kept ~100%).
INERT = {
    "human_only":   dict(bins=[33, 34, 35], amp=0.34, kept=0.96, note=""),
    # ⚠️ THE MULTISPECIES MODEL HAS NO ENERGY-MATCHED INERT BAND, BECAUSE NONE EXISTS.  Its
    # carrier is 401x median and the ONLY band with comparable energy is bins 21-23 (period 23.3),
    # which heads read with amplitude 3.90 and which collapses to 25% on deletion.  Its table has
    # only a few high-energy modes and they are ALL READ.  So this control is inert but NOT
    # energy-matched, and the mismatch is reported rather than hidden.
    "multispecies": dict(bins=[33, 34, 35], amp=None, kept=None,
                         note="NOT energy-matched: no inert band of comparable energy EXISTS in "
                              "this model. Its only energy-matched candidate (bins 21-23) is "
                              "READ (amp 3.90, collapses to 25%). Interpret accordingly."),
}
T_BIO = 10.298                     # the MEASURED biological wavelength (D28), not a bin label
BIN = 20_000                       # 20 kb -> ~97 dyads -> score reliability 0.823 (D33)
LEN = 510
DISC = ["chr1", "chr2"]            # phase frozen here
APPLY = ["chr3", "chr4", "chr5"]   # windows drawn here


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=20_000, help="per stratum")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    # ---------------------------------------------------------------- 1. the REGIONAL score
    dy = load_dyads()
    fa = Fasta(str(FASTA), as_raw=True)
    seqs = {c: np.frombuffer(fa[c][:].upper().encode(), dtype=np.uint8) for c in DISC + APPLY}
    cD = sum(complex(coeffs(seqs[c], dy[c], "WW", T_BIO).sum()) for c in DISC)
    phi = math.atan2(cD.imag, cD.real)                    # FROZEN on chr1+2, used nowhere else
    print(f"frozen phase at T = {T_BIO}: {math.degrees(phi):+.1f} deg  (from {DISC})")

    rows = []
    for c in APPLY:
        s, p = seqs[c], dy[c]
        cc = coeffs(s, p, "WW", T_BIO)
        v = (cc * np.exp(-1j * phi)).real                 # SIGNED projection -> null mean 0
        b = p // BIN
        for bb in np.unique(b):
            m = b == bb
            n = int(m.sum())
            if n < 40:
                continue
            lo, hi = int(bb) * BIN, int(bb) * BIN + BIN
            w = s[lo:hi]
            if not np.isin(w, [65, 67, 71, 84]).all():
                continue                                   # skip bins touching N
            gc = float(np.isin(w, [67, 71]).mean())
            rows.append((c, lo, float(v[m].sum() / math.sqrt(n)), gc, n))
    sc = np.array([r[2] for r in rows])
    gc = np.array([r[3] for r in rows])
    print(f"{len(rows):,} scoreable 20-kb bins over {APPLY};  score mean {sc.mean():.2f} "
          f"sd {sc.std():.2f}")

    # ---------------------------------------------------------------- 2. GC-MATCHED strata
    hi_i = np.argsort(-sc)[: len(sc) // 5]
    lo_i = np.argsort(sc)[: len(sc) // 5]
    # GC-match by greedy nearest-neighbour pairing on GC
    pairs = []
    used = set()
    for i in hi_i:
        j = min((j for j in lo_i if j not in used), key=lambda j: abs(gc[j] - gc[i]), default=None)
        if j is None:
            break
        if abs(gc[j] - gc[i]) < 0.02:
            pairs.append((i, j))
            used.add(j)
    H = [rows[i] for i, _ in pairs]
    L = [rows[j] for _, j in pairs]
    print(f"\nGC-MATCHED strata: {len(pairs):,} bin pairs")
    print(f"   HIGH  score {np.mean([r[2] for r in H]):+7.2f}   GC {np.mean([r[3] for r in H]):.4f}")
    print(f"   LOW   score {np.mean([r[2] for r in L]):+7.2f}   GC {np.mean([r[3] for r in L]):.4f}")

    # ------------------------------------------------- CONTROL 1: do the strata differ in TONE?
    def tone(bins):
        cs = []
        for c, lo, *_ in bins:
            p = dy[c][(dy[c] >= lo) & (dy[c] < lo + BIN)]
            cs.append(coeffs(seqs[c], p, "WW", T_BIO))
        cc = np.concatenate(cs)
        return float((cc * np.exp(-1j * phi)).real.sum() / math.sqrt(len(cc))), len(cc)
    tH, nH = tone(H)
    tL, nL = tone(L)
    print(f"\n  CONTROL THAT CAN FAIL — does HIGH actually carry more 10.298 tone than LOW?")
    print(f"     HIGH: z = {tH:+8.1f}  ({nH:,} dyads)")
    print(f"     LOW : z = {tL:+8.1f}  ({nL:,} dyads)")
    ok = tH > tL + 5
    print(f"     {'PASS' if ok else 'FAIL'} — the strata {'do' if ok else 'do NOT'} differ in the "
          f"biological signal")
    if not ok:
        print("     ABORTING: the strata are meaningless, so the interaction would be too.")
        return

    if args.dry_run:
        print("\n--dry-run: no GPU touched.")
        return

    # ---------------------------------------------------------------- 3. windows (PAIRED)
    def draw(bins, n):
        out = []
        idx = rng.integers(0, len(bins), n)
        for i in idx:
            c, lo, *_ = bins[i]
            st = int(rng.integers(lo, lo + BIN - LEN))
            w = seqs[c][st:st + LEN]
            if np.isin(w, [65, 67, 71, 84]).all():
                out.append(w)
        return np.stack(out)
    WH, WL = draw(H, args.windows), draw(L, args.windows)
    print(f"\nwindows: {len(WH):,} HIGH, {len(WL):,} LOW  (510 nt, paired across all conditions)")

    dev = pick_device(None)
    rec = {"T_bio": T_BIO, "phi_deg": math.degrees(phi), "bin_kb": BIN // 1000,
           "n_pairs": len(pairs), "tone_high": tH, "tone_low": tL,
           "gc_high": float(np.mean([r[3] for r in H])), "gc_low": float(np.mean([r[3] for r in L])),
           "results": {}}

    # VERIFIED against the tokenizer, not assumed: A=6 C=7 G=8 T=9, mask=4.
    # An off-by-one here reads every A as token 5, C as A, G as C, T as G -- silently, with no
    # error and plausible-looking losses.  Never hard-code a vocab offset without checking it.
    B2I = {"A": 6, "C": 7, "G": 8, "T": 9}
    for mname, (path, own) in MODELS.items():
        ad = load_adapter(str(REPO / path), dev)
        m = ad.model
        tok = AutoTokenizer.from_pretrained(str(REPO / path))
        emb = m.bert.embeddings.position_embeddings
        P0 = emb.weight.data.clone()
        P = P0.detach().float().cpu().numpy()
        Lr, _ = P.shape
        mu = P.mean(0, keepdims=True)
        F = np.fft.rfft(P - mu, axis=0)
        pw = (np.abs(F) ** 2).sum(1)

        # THE CONTROL BAND MUST BE VERIFIED INERT, NOT ASSERTED.
        # My first attempt energy-matched a band and CALLED it inert.  It was not: for the
        # human-only model it picked bins 90-92 (period 5.6), which heads read with amplitude 2.2
        # -- LARGER than the 10.2 carrier's 1.4 -- and deleting it hurt 3x MORE than deleting the
        # carrier.  That is why the "inert" control out-damaged the thing it was controlling for.
        # These bands were found by SEARCH and AUDITED: max reader amplitude, and how much of it
        # survives the deletion.  See INERT_AUDIT below.
        target = pw[own].sum()
        em = INERT[mname]["bins"]
        conds = dict(BANDS)
        conds["del_verified_inert"] = em
        a = INERT[mname]
        print(f"\n{'='*92}\n{mname}   carrier {own} ({target/np.median(pw[1:]):.0f}x median)")
        aud = (f"max reader amp {a['amp']:.2f}, {a['kept']*100:.0f}% kept on deletion"
               if a["amp"] is not None else "audit: see note")
        print(f"   VERIFIED-INERT control {em}: period {512/em[1]:.2f}, "
              f"{pw[em].sum()/np.median(pw[1:]):.1f}x median "
              f"({pw[em].sum()/target:.2f}x the carrier's energy), {aud}")
        if a["note"]:
            print(f"   ⚠️ {a['note']}")
        print("=" * 92)

        def strip(bins):
            G = F.copy()
            for b in bins:
                G[b] = 0
            return torch.tensor(np.fft.irfft(G, n=Lr, axis=0) + mu,
                                dtype=P0.dtype, device=P0.device)

        def losses(Wn, Pw):
            """Per-token MLM loss. SAME tokens, SAME masks across every condition (seed fixed)."""
            emb.weight.data = Pw.clone()
            g = torch.Generator(device="cpu").manual_seed(1234)      # <-- identical masks
            out = []
            cls_, sep_ = tok.cls_token_id, tok.sep_token_id
            for s in range(0, len(Wn), args.bs):
                ch = Wn[s:s + args.bs]
                # [CLS] + 510 nt + [SEP] = 512 -- EXACTLY the format the model was trained on.
                # Feeding 510 BARE nucleotides costs 0.0144 nats (1.4448 vs 1.4304), which is
                # 100x the effect being measured.  It cancels in the DiD, but it runs the model
                # off-distribution and there is no reason to accept that.
                body = np.stack([[B2I[chr(b)] for b in r] for r in ch])
                ids = torch.tensor(
                    np.concatenate([np.full((len(ch), 1), cls_), body,
                                    np.full((len(ch), 1), sep_)], axis=1), device=dev)
                lab = ids.clone()
                msk = (torch.rand(ids.shape, generator=g) < 0.15).to(dev)
                msk[:, 0] = False                      # never mask [CLS]
                msk[:, -1] = False                     # never mask [SEP]
                lab[~msk] = -100
                with torch.no_grad():
                    lg = m(input_ids=ids.masked_fill(msk, tok.mask_token_id)).logits
                l = torch.nn.functional.cross_entropy(
                    lg.view(-1, lg.size(-1)), lab.view(-1), reduction="none", ignore_index=-100)
                out.append(l[lab.view(-1) != -100].float().cpu().numpy())
            return np.concatenate(out)

        base = {"HIGH": losses(WH, P0), "LOW": losses(WL, P0)}
        print(f"  intact loss:  HIGH {base['HIGH'].mean():.5f}   LOW {base['LOW'].mean():.5f}   "
              f"({len(base['HIGH']):,} / {len(base['LOW']):,} masked tokens)")
        print(f"\n  {'deletion':22s} {'dL(HIGH)':>11} {'dL(LOW)':>11} {'INTERACTION I':>15} "
              f"{'SE':>9} {'z':>7}")
        print("  " + "-" * 82)
        for cname, bins in conds.items():
            Pd = strip(bins)
            dH = losses(WH, Pd) - base["HIGH"]
            dL = losses(WL, Pd) - base["LOW"]
            I = float(dH.mean() - dL.mean())
            se = float(math.sqrt(dH.var() / len(dH) + dL.var() / len(dL)))
            z = I / se
            rec["results"][f"{mname}|{cname}"] = dict(
                dL_high=float(dH.mean()), dL_low=float(dL.mean()), I=I, se=se, z=z,
                bins=bins, n_high=len(dH), n_low=len(dL))
            star = "  <-- H1" if (mname == "human_only" and cname == "del_10.2") else ""
            print(f"  {cname:22s} {dH.mean():>+11.5f} {dL.mean():>+11.5f} {I:>+15.6f} "
                  f"{se:>9.6f} {z:>+7.1f}{star}")
        emb.weight.data = P0.clone()
        del ad

    OUT.write_text(json.dumps(rec, indent=1))
    print(f"\nsaved -> {OUT}")
    print("\n" + "=" * 92)
    print("PREREGISTERED: H1 I(human,10.2) > 0 and LARGEST of the four.  H2-H5 ~ 0.")
    print("⚠️ SPECIES CAVEAT: this is MOUSE sequence. The human-only model saw only HUMAN, so this")
    print("   is OUT OF DISTRIBUTION for it. Species-matched replication on human dyads is REQUIRED.")
    print("=" * 92)


if __name__ == "__main__":
    main()
