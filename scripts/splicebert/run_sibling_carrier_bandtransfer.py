#!/usr/bin/env python
"""IS THE PERIOD-3 CARRIER USED IN THE STAGE-1 72-SPECIES SIBLING?  Band-deletion transfer test.

Companion to run_carrier_surgery.py (final 1024nt checkpoint). This runs the SAME 2x2
band-deletion design on the stage-1 72-species sibling SpliceBERT.510nt -- the checkpoint where
the P-spectrum inventory (results/origin-siblings) found the period-3 carrier present (250.7x at
bin 171). The human-only sibling is NOT tested here: its period-3 carrier is already ABSENT in P
(6.2x, undetected), so there is nothing to delete.

TERMINOLOGY: this is a Fourier BAND-DELETION / carrier-TRANSFER intervention on the position
embeddings -- delete the period-3 band from P, transfer the deleted-P model forward, and measure
what it can no longer predict. Energy-matched control bands are the null.

512-LENGTH ADAPTATION. N=512 is not divisible by 3, so period-3 is not a single exact bin; its
energy smears across bins 170/171/172 (peak 250.7x at 171). The band here is [169..173] (5 bins),
chosen to capture the smeared period-3 energy; control bands use the same width.

DESIGN (2x2 with an empirical null)
    treatment : delete P's period-3 band (169-173)
    control   : delete each of N energy-matched bands elsewhere (same width, power within 25%,
                away from period 3 / 1.5 / 6)
    regions   : CDS-exon interior  vs  intron interior
    readout   : mean log-prob of the TRUE base at masked positions
    damage(arm, region) = LL(intact, region) - LL(arm, region)
    DiD(arm)            = damage(arm, exon) - damage(arm, intron)
HEALTH GATE: an arm that keeps < 50% of intact headroom over chance is not interpreted (R13).
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
    DEFAULT_FASTA, DEFAULT_GTF, CouplingTask, eligible, gboot, load_adapter, pick_device,
    _dataset, stable_rng,
)

BASE = "data/models/splicebert_official_archive/extracted/models"
BAND = list(range(169, 174))          # period-3 carrier + smeared sidebands at N=512
CHANCE = float(np.log(0.25))
MASK_RATE = 1 / 7
MIN_LEN = 60


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--controls", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=f"{BASE}/SpliceBERT.510nt")
    ap.add_argument("--out", default="results/origin-siblings/canonical/sibling_carrier_bandtransfer.json")
    args = ap.parse_args()

    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=True)
    task = CouplingTask(ad, batch=args.batch)
    OFF = task.off
    maxnt = ad.max_nucleotides

    emb = ad.model.bert.embeddings.position_embeddings
    P0 = emb.weight.detach().clone()
    P = P0.float().cpu().numpy()
    N, _ = P.shape
    mu = P.mean(0, keepdims=True)
    F = np.fft.rfft(P - mu, axis=0)
    pw = (np.abs(F) ** 2).sum(1)
    W = len(BAND)
    target = float(pw[BAND].sum())

    def make(bins):
        G = F.copy()
        for b in bins:
            G[b] = 0
        return torch.tensor(np.fft.irfft(G, n=N, axis=0) + mu,
                            dtype=P0.dtype, device=P0.device)

    # energy-matched control bands: same width, power within 25%, away from period 3/1.5/6
    cand = []
    for lo in range(3, len(pw) - W):
        b = list(range(lo, lo + W))
        per = N / np.mean(b)
        if abs(per - 3.0) < 0.6 or abs(per - 1.5) < 0.3 or abs(per - 6.0) < 0.6:
            continue
        e = float(pw[b].sum())
        if 0.75 * target <= e <= 1.25 * target:
            cand.append((b, e))
    rng = np.random.default_rng(args.seed)
    if len(cand) > args.controls:
        idx = rng.choice(len(cand), args.controls, replace=False)
        cand = [cand[i] for i in idx]
    print(f"model {args.model}")
    print(f"period-3 band {BAND[0]}-{BAND[-1]} (N={N})  power {target:.0f}")
    print(f"{len(cand)} energy-matched control bands (width {W}, power within 25%)\n")

    # build cohort directly (no coupling forward -> no >maxnt overflow).
    # Sequences are fixed 1024-nt windows; the 510 sibling caps at maxnt, so CROP each window to
    # [acceptor-PAD, next_acceptor+PAD] (the exon+intron span of interest) and shift positions.
    from dataclasses import replace as _replace
    PAD = 40
    Xall = eligible(_dataset(args.gtf, args.fasta), args.split)
    stable_rng(f"sibling-bandtransfer-{args.split}", 0).shuffle(Xall)

    def crop(x):
        L = max(0, x.acceptor_pos - PAD)
        R = min(len(x.sequence), x.next_acceptor_pos + PAD)
        if (R - L) + OFF + 1 > maxnt:
            return None
        return _replace(x, sequence=x.sequence[L:R],
                        acceptor_pos=x.acceptor_pos - L, donor_pos=x.donor_pos - L,
                        next_acceptor_pos=x.next_acceptor_pos - L)

    X = []
    for x in Xall:
        if not (x.gene_type == "protein_coding" and x.exon_length >= MIN_LEN
                and (x.next_acceptor_pos - x.donor_pos) >= MIN_LEN):
            continue
        xc = crop(x)
        if xc is not None:
            X.append(xc)
        if len(X) >= args.n:
            break
    print(f"{len(X)} protein-coding exons (cropped len<= {maxnt}), "
          f"{len(set(x.gene_id for x in X))} genes   device={dev}")

    mrng = np.random.default_rng(args.seed)
    MASKS = [mrng.random(len(x.sequence)) < MASK_RATE for x in X]

    def regions(x):
        return {"exon": (x.acceptor_pos + 6, x.donor_pos - 6),
                "intron": (x.donor_pos + 8, x.next_acceptor_pos - 8)}

    def evaluate() -> dict:
        out: dict = {"exon": [], "intron": []}
        genes: dict = {"exon": [], "intron": []}
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
                            genes[r].append(x.gene_id)
        return {"ll": out, "genes": genes}

    ARMS: list[tuple[str, list[int]]] = [("intact", []), ("period3", BAND)]
    ARMS += [(f"ctrl{i}", b) for i, (b, _) in enumerate(cand)]

    res: dict = {}
    for k, (name, bins) in enumerate(ARMS):
        emb.weight.data = P0.clone() if not bins else make(bins)
        res[name] = evaluate()
        m = {reg: float(np.mean(res[name]["ll"][reg])) for reg in ("exon", "intron")}
        print(f"  [{k+1}/{len(ARMS)}] {name:9s} exon {m['exon']:+.4f}  intron {m['intron']:+.4f}",
              end="\r", flush=True)
    emb.weight.data = P0.clone()
    print()

    base = {r: float(np.mean(res["intact"]["ll"][r])) for r in ("exon", "intron")}
    hr0 = {r: base[r] - CHANCE for r in base}
    genes = res["intact"]["genes"]

    def summarise(name):
        ll = {r: float(np.mean(res[name]["ll"][r])) for r in ("exon", "intron")}
        dmg = {r: base[r] - ll[r] for r in ll}
        kept = {r: (ll[r] - CHANCE) / hr0[r] for r in ll}
        return dict(ll=ll, damage=dmg, kept=kept, did=dmg["exon"] - dmg["intron"],
                    healthy=bool(min(kept.values()) >= 0.5))

    S = {n: summarise(n) for n, _ in ARMS if n != "intact"}
    p3 = S["period3"]
    ctrls = [S[n] for n in S if n.startswith("ctrl")]

    cd = np.array([c["did"] for c in ctrls])
    ce = np.array([c["damage"]["exon"] for c in ctrls])
    p_did = float((1 + np.sum(cd >= p3["did"])) / (1 + len(cd)))
    p_exon = float((1 + np.sum(ce >= p3["damage"]["exon"])) / (1 + len(ce)))

    de = np.array(res["intact"]["ll"]["exon"]) - np.array(res["period3"]["ll"]["exon"])
    di = np.array(res["intact"]["ll"]["intron"]) - np.array(res["period3"]["ll"]["intron"])
    lo_e, hi_e = gboot(de, genes["exon"])
    lo_i, hi_i = gboot(di, genes["intron"])

    print("=" * 88)
    print("SIBLING CARRIER BAND-DELETION (stage-1 72-species) -- is period-3 used for coding?")
    print(f"  intact:  exon {base['exon']:+.4f}   intron {base['intron']:+.4f}  [chance {CHANCE:+.4f}]")
    print(f"  period-3 arm keeps {p3['kept']['exon']:.1%} exon / {p3['kept']['intron']:.1%} intron headroom "
          f"{'OK' if p3['healthy'] else '*** BELOW FLOOR ***'}")
    print(f"  PERIOD-3   exon dmg {p3['damage']['exon']:+.4f}  intron dmg {p3['damage']['intron']:+.4f}  DiD {p3['did']:+.4f}")
    print(f"  controls   exon dmg {ce.mean():+.4f} (max {ce.max():+.4f})  DiD mean {cd.mean():+.4f} (max {cd.max():+.4f})")
    print(f"  add-one p (DiD) = {p_did:.4f}   add-one p (exon dmg) = {p_exon:.4f}")
    print(f"  exon damage {de.mean():+.4f} [{lo_e:+.4f},{hi_e:+.4f}]  intron {di.mean():+.4f} [{lo_i:+.4f},{hi_i:+.4f}] (gene-grouped)")

    verdict = ("VOID_unhealthy" if not p3["healthy"]
               else "USED_coding_specific" if (p_did < 0.05 and p3["did"] > 0)
               else "USED_not_coding_specific" if p_exon < 0.05
               else "NOT_USED")
    print(f"  VERDICT: {verdict}")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "origin-siblings-carrier-bandtransfer-v1",
        "model": args.model, "corpus": "72-species-stage1", "n": len(X), "N": N,
        "band": BAND, "band_power": target, "chance": CHANCE,
        "intact": base, "headroom": hr0,
        "period3": p3,
        "controls": [dict(bins=b, **S[f"ctrl{i}"]) for i, (b, _) in enumerate(cand)],
        "p_did": p_did, "p_exon": p_exon,
        "exon_damage_ci": [lo_e, hi_e], "intron_damage_ci": [lo_i, hi_i],
        "verdict": verdict,
    }
    out.write_text(json.dumps(payload, indent=1))
    ExperimentManifest.create(
        "origin_siblings:carrier_bandtransfer", sys.argv,
        config={"model": args.model, "n": len(X), "band": BAND, "controls": len(cand)},
        notes=["Band-deletion transfer on the stage-1 72-species sibling (SpliceBERT.510nt), the "
               "checkpoint where the P-spectrum inventory found period-3 present (250.7x, bin 171).",
               "N=512 is not divisible by 3, so period-3 smears across bins 170-172; the band is "
               "169-173 (5 bins). Energy-matched control bands are the null.",
               "Human-only sibling not tested: its period-3 carrier is already absent in P.",
               "MLM health gated: an arm below 50% headroom over chance is not interpreted (R13)."],
    ).write(out.parent / "sibling_carrier_bandtransfer.manifest.json")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
