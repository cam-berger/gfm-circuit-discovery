#!/usr/bin/env python
"""DOSE-RESPONSE ON THE PERIOD-3 CARRIER -- and an INDEPENDENT health check.

THE RESULT THIS IS TESTING.  Deleting P's period-3 band (7 of 513 Fourier bins) costs SpliceBERT
0.1743 nats of masked-base log-prob in CODING EXONS and only 0.0428 in INTRONS.  Twenty-one
energy-matched control bands cost 0.0065 (mean) / 0.0265 (worst) in exons.  The DiD is 0.1315
against a control maximum of 0.0037 -- a factor of 35.

TWO THINGS ARE WRONG WITH STOPPING THERE.

1. THE HEALTH GATE I WROTE MISFIRED, AND IT MISFIRED BY CONDITIONING ON THE OUTCOME.
   It required the arm to keep >= 50% of its EXON headroom -- but exon prediction IS the outcome.
   A gate that trips whenever the effect is large cannot tell "the model is broken" from "the
   model lost exactly the ability under test."  That is the R10 error (conditioning on the
   outcome), wearing R13's clothes.

   The health check must be on a behaviour the mechanism is NOT claimed to serve.  Two are used
   here, and neither is the outcome:
       (a) INTRON masked-base prediction   -- same readout, no codon structure.  (83.5% retained)
       (b) THE DONOR-ACCEPTOR COUPLING     -- a completely different behaviour, the subject of
           the other study: margin(clean) - margin(partner AG destroyed), mediated by L3H1.
           If the coupling survives the surgery, the model is demonstrably still working.

2. A BINARY DELETION IS ONE POINT.  A DOSE-RESPONSE IS A CURVE.
   Attenuate the carrier by alpha in {1.0, 0.9, 0.75, 0.5, 0.25, 0.1, 0.0} instead of deleting
   it.  If exon damage rises monotonically as the carrier is weakened -- and intron damage does
   not -- that is a dose-response, and no single-point artefact survives it.  It also finds an
   alpha at which the model is unambiguously healthy AND the effect is already visible, which
   settles the health question on its own terms.

   MATCHED CONTROL: attenuate the WORST-CASE control band (the energy-matched band that did the
   most exon damage when deleted) by the same alphas.  Its curve is the null.
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

from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    CORRUPT_ALT, DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, gboot, load_adapter,
    load_examples, pick_device,
)
from gfmcd.splicing.coupling import substitute  # noqa: E402

BAND = list(range(339, 346))
ALPHAS = (1.0, 0.9, 0.75, 0.5, 0.25, 0.1, 0.0)
CHANCE = float(np.log(0.25))
MASK_RATE = 1 / 7
MIN_LEN = 60


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--surgery", default="results/phase-lattice/carrier_surgery.json")
    ap.add_argument("--out", default="results/phase-lattice/carrier_dose.json")
    args = ap.parse_args()

    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=True)
    task = CouplingTask(ad, batch=args.batch)
    OFF = task.off

    emb = ad.model.bert.embeddings.position_embeddings
    P0 = emb.weight.detach().clone()
    P = P0.float().cpu().numpy()
    N, _ = P.shape
    mu = P.mean(0, keepdims=True)
    F = np.fft.rfft(P - mu, axis=0)

    # the WORST control band from the surgery run -- the hardest null available
    SG = json.loads((REPO / args.surgery).read_text())
    worst = max(SG["controls"], key=lambda c: c["damage"]["exon"])
    CTRL = worst["bins"]
    print(f"treatment band {BAND[0]}-{BAND[-1]}  (period ~3)")
    print(f"control   band {CTRL[0]}-{CTRL[-1]}  (period ~{N/np.mean(CTRL):.1f}) -- the "
          f"energy-matched band that did the MOST exon damage when deleted "
          f"({worst['damage']['exon']:.4f})\n")

    def scaled(bins, alpha):
        G = F.copy()
        for b in bins:
            G[b] = G[b] * alpha
        return torch.tensor(np.fft.irfft(G, n=N, axis=0) + mu,
                            dtype=P0.dtype, device=P0.device)

    X, _ = load_examples(args.gtf, args.fasta, args.split, args.n * 4, task=task,
                         tag=f"dose-{args.split}", min_effect=None)
    X = [x for x in X if x.gene_type == "protein_coding" and x.exon_length >= MIN_LEN
         and x.next_acceptor_pos - x.donor_pos >= MIN_LEN][:args.n]
    genes = [x.gene_id for x in X]
    print(f"{len(X)} protein-coding exons, {len(set(genes))} genes   device={dev}\n")

    mrng = np.random.default_rng(args.seed)
    MASKS = [mrng.random(len(x.sequence)) < MASK_RATE for x in X]

    def regions(x):
        return {"exon": (x.acceptor_pos + 6, x.donor_pos - 6),
                "intron": (x.donor_pos + 8, x.next_acceptor_pos - 8)}

    RG = {r: [x for x in X if regions(x)[r][1] - regions(x)[r][0] >= 24]
          for r in ("exon", "intron")}
    RGG = {r: [x.gene_id for x in RG[r]] for r in RG}

    def mlm() -> dict:
        out: dict = {"exon": [], "intron": []}
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
                ll = torch.log_softmax(ad.model(**enc).logits.float(), dim=-1)
                for j, x in enumerate(ch):
                    idx = torch.tensor([p + OFF for p in pos[j]], device=ll.device)
                    tgt = torch.tensor([task.bid[x.sequence[p]] for p in pos[j]],
                                       device=ll.device)
                    v = np.full(len(x.sequence), np.nan)
                    for p, q in zip(pos[j], ll[j, idx, tgt].cpu().numpy()):
                        v[p] = q
                    for r, (a, b) in regions(x).items():
                        if b - a >= 24:
                            out[r].append(float(np.nanmean(v[a:b])))
        return out

    def coupling() -> list[float]:
        """THE INDEPENDENT HEALTH CHECK: a behaviour the carrier is NOT claimed to serve.

        margin(clean) - margin(partner's AG destroyed), read at the donor's masked G.
        This is the other study's entire subject, mediated by L3H1 <- L2H15.
        """
        out = []
        with torch.no_grad():
            for i in range(0, len(X), task.batch):
                ch = X[i:i + task.batch]
                dn = [x.donor_pos for x in ch]
                cl = [x.sequence for x in ch]
                co = [substitute(x.sequence, x.next_acceptor_pos - 2, CORRUPT_ALT) for x in ch]
                m_cl = task.margin(ad.model(**task.encode(cl, dn)).logits.float(), dn)
                m_co = task.margin(ad.model(**task.encode(co, dn)).logits.float(), dn)
                out += (m_cl - m_co).tolist()
        return out

    rows = []
    for bins, name in ((BAND, "period-3"), (CTRL, "control")):
        for a in ALPHAS:
            if a == 1.0 and name == "control":
                continue     # alpha=1 is the intact model; already measured. (`break` here was a
                             # bug: it skipped the ENTIRE control dose-response, i.e. the null.)
            emb.weight.data = P0.clone() if a == 1.0 else scaled(bins, a)
            L = mlm()
            C = coupling()
            rows.append(dict(band=name, alpha=a,
                             exon=float(np.mean(L["exon"])), intron=float(np.mean(L["intron"])),
                             coupling=float(np.mean(C)),
                             exon_all=L["exon"], intron_all=L["intron"], coup_all=C))
            print(f"  {name:9s} alpha={a:4.2f}  exon {np.mean(L['exon']):+.4f}  "
                  f"intron {np.mean(L['intron']):+.4f}  coupling {np.mean(C):+.4f}")
    emb.weight.data = P0.clone()

    base = next(r for r in rows if r["alpha"] == 1.0)
    hr0 = {r: base[r] - CHANCE for r in ("exon", "intron")}

    print("\n" + "=" * 94)
    print("DOSE-RESPONSE: attenuate the period-3 carrier by alpha (1.0 = intact, 0.0 = deleted)")
    print(f"  intact: exon {base['exon']:+.4f}  intron {base['intron']:+.4f}  "
          f"coupling {base['coupling']:+.4f}  [chance {CHANCE:+.4f}]")
    print(f"\n  {'band':9s} {'alpha':>5s} {'exon dmg':>9s} {'intron dmg':>11s} {'DiD':>8s} "
          f"{'intron kept':>12s} {'COUPLING':>9s} {'coup kept':>10s}")
    for r in rows:
        if r["alpha"] == 1.0 and r["band"] == "control":
            continue
        de = base["exon"] - r["exon"]
        di = base["intron"] - r["intron"]
        ik = (r["intron"] - CHANCE) / hr0["intron"]
        ck = r["coupling"] / base["coupling"]
        print(f"  {r['band']:9s} {r['alpha']:5.2f} {de:9.4f} {di:11.4f} {de-di:8.4f} "
              f"{ik:11.1%} {r['coupling']:9.4f} {ck:9.1%}")

    p3 = [r for r in rows if r["band"] == "period-3"]
    ct = [r for r in rows if r["band"] == "control"]
    de3 = [base["exon"] - r["exon"] for r in p3]
    dec = [base["exon"] - r["exon"] for r in ct]
    mono = all(de3[i] <= de3[i + 1] + 1e-4 for i in range(len(de3) - 1))

    print("\n  MONOTONIC dose-response in exon damage: "
          f"{'YES' if mono else 'NO'}   {[round(d,4) for d in de3]}")
    print(f"  same for the control band:               {[round(d,4) for d in dec]}")

    # THE CONTROL THAT DECIDES THE COUPLING RESULT.  Attenuating the carrier perturbs P
    # GLOBALLY, and the coupling is a far more delicate behaviour than masked-base prediction.
    # A modest global perturbation could kill a delicate circuit while leaving a robust one
    # standing.  So: does the energy-matched control band kill the coupling too?
    print("\n  COUPLING vs CARRIER STRENGTH -- period-3 band against the energy-matched null:")
    print(f"    {'alpha':>5s} {'period-3':>10s} {'control':>10s}")
    for a in ALPHAS:
        r3 = next((r for r in p3 if r["alpha"] == a), None)
        rc = next((r for r in ct if r["alpha"] == a), None)
        c3 = f"{r3['coupling']/base['coupling']:9.1%}" if r3 else "        --"
        cc = f"{rc['coupling']/base['coupling']:9.1%}" if rc else "  (=100%)"
        print(f"    {a:5.2f} {c3:>10s} {cc:>10s}")

    # the honest health statement, at the LARGEST alpha where the effect is already clear
    print("\n  INDEPENDENT HEALTH (neither is the outcome):")
    for r in p3:
        if r["alpha"] in (0.5, 0.25):
            ik = (r["intron"] - CHANCE) / hr0["intron"]
            ck = r["coupling"] / base["coupling"]
            de = base["exon"] - r["exon"]
            print(f"    alpha={r['alpha']:.2f}: intron headroom {ik:.1%}, coupling {ck:.1%} of "
                  f"baseline  ->  exon damage still {de:.4f}")

    # gene-grouped CI on the DiD at full deletion
    r0 = next(r for r in p3 if r["alpha"] == 0.0)
    de = np.array(base["exon_all"]) - np.array(r0["exon_all"])
    di = np.array(base["intron_all"]) - np.array(r0["intron_all"])
    lo_e, hi_e = gboot(de, RGG["exon"])
    lo_i, hi_i = gboot(di, RGG["intron"])
    print(f"\n  at alpha=0:  exon damage {de.mean():+.4f} [{lo_e:+.4f},{hi_e:+.4f}]   "
          f"intron {di.mean():+.4f} [{lo_i:+.4f},{hi_i:+.4f}]  (gene-grouped)")

    out = REPO / args.out
    out.write_text(json.dumps({
        "n": len(X), "band": BAND, "control_band": CTRL, "alphas": list(ALPHAS),
        "chance": CHANCE, "monotonic": bool(mono),
        "rows": [{k: v for k, v in r.items() if not k.endswith("_all")} for r in rows],
        "exon_damage_ci": [lo_e, hi_e], "intron_damage_ci": [lo_i, hi_i],
    }, indent=1))
    ExperimentManifest.create(
        "phase_lattice:carrier_dose", sys.argv,
        config={"n": len(X), "band": BAND, "control": CTRL, "alphas": list(ALPHAS)},
        notes=["The health gate in run_carrier_surgery.py CONDITIONED ON THE OUTCOME (it gated "
               "on exon headroom, which is the outcome). Health is checked here on two "
               "behaviours the carrier is not claimed to serve: intron MLM, and the "
               "donor-acceptor coupling.",
               "A dose-response curve cannot be produced by a single-point artefact."],
    ).write(out.parent / "carrier_dose.manifest.json")
    print("=" * 94)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
