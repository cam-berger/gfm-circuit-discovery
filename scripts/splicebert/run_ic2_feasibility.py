#!/usr/bin/env python
"""IC2 feasibility pilot (TRAIN-only, resource-capped). Answers the go/no-go questions for a SYNTHETIC
two-candidate acceptor-competition assay (natural dual-acceptor yield is inadequate: 142 loci, 0 at >=80nt).
Does NOT build IC2; does NOT read val. Behaviour-level.

Q1 readout stability: sealed margin f=z_G-max(z_A,z_C,z_T) is PIECEWISE-LINEAR -- nonlinear only when the
   non-G argmax SWITCHES across the 2x2 corruption cells. Measure the switch rate + fixed-class-margin
   agreement (z_G-z_A, z_G-z_C, z_G-z_T).
Q2 viability: do BOTH candidates (native P + inserted decoy D) contribute measurably; native-anchor + OOD intact.
Q3 ranking-vs-control: does a real acceptor D contribute MORE than a matched non-acceptor AG sham at the
   same slot (acceptor-specific support beyond generic AG damage)?
MDE: between-locus SD of D reliance -> expected census MDE.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts" / "splicebert"))
from maxentpy import maxent                                                       # noqa: E402
from maxentpy.maxent import load_matrix3                                          # noqa: E402
from gfmcd.splicing.circuit import (                                             # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, load_adapter, pick_device, eligible, _dataset)
from gfmcd.splicing.coupling import substitute                                    # noqa: E402
from run_ic1_pilot import ood_nll                                                 # noqa: E402
M3 = load_matrix3()


def shash(g): return int(hashlib.sha256(str(g).encode()).hexdigest()[:8], 16)


def logits_at_donor(ad, task, seqs, donors, batch):
    """return per-seq (f_sealed, argmax_nonG in {A,C,T}, [zG-zA, zG-zC, zG-zT])."""
    off, bid = task.off, task.bid
    outs = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            ss, dd = seqs[i:i + batch], donors[i:i + batch]
            enc = task.encode(ss, dd)
            lg = ad.model(**enc).logits.float()
            for j, d in enumerate(dd):
                v = lg[j, d + off]
                zG, zA, zC, zT = float(v[bid["G"]]), float(v[bid["A"]]), float(v[bid["C"]]), float(v[bid["T"]])
                nonG = {"A": zA, "C": zC, "T": zT}
                am = max(nonG, key=nonG.get)
                outs.append((zG - max(zA, zC, zT), am, [zG - zA, zG - zC, zG - zT]))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pilot", type=int, default=50)
    ap.add_argument("--batch", type=int, default=12)          # modest (shared laptop)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF); ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="results/splicebert_coupling/development/ic2_feasibility.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    dev = pick_device(args.device); ad = load_adapter(args.model, dev); task = CouplingTask(ad, batch=args.batch)

    tr = eligible(_dataset(args.gtf, args.fasta), "train")
    seen, dd = set(), []
    for x in tr:
        if x.gene_id not in seen:
            seen.add(x.gene_id); dd.append(x)
    # need room for donor(protected) + inserted D (40nt, AG 60nt upstream of P) + native P
    def ok(x):
        a, d0 = x.next_acceptor_pos, x.donor_pos
        aD = a - 60
        return (a - d0) >= 110 and (aD - 37) > d0 + 8 and (aD + 3) < a - 25
    pool = [x for x in dd if ok(x)]
    # frozen decoy pool: real acceptor 40mers [acc-37..acc+3] from OTHER genes (score3-scored)
    def acc40(x):
        a = x.next_acceptor_pos
        return x.sequence[a - 37:a + 3] if a - 37 >= 0 else None
    decoys = [(acc40(x), x.gene_id) for x in pool if acc40(x) and len(acc40(x)) == 40]
    rng = np.random.default_rng(0)
    pilot = pool[:args.n_pilot]
    print(f"train dedup={len(dd)} pool(len>=110, slot-viable)={len(pool)} decoys={len(decoys)} n_pilot={len(pilot)}")

    def sham40(comp_src):
        """non-acceptor 40mer: shuffle a real window's bases (kills score3) but keep the AG at [35,36]
        so it lands at [aD-2,aD-1] after insertion -- matching D's AG position AND the aD-2 corruption."""
        b = list(comp_src); rng.shuffle(b); b[35], b[36] = "A", "G"; return "".join(b)

    seqs, dons, tags = [], [], []
    owins, oseqs, otags = [], [], []
    meta = []
    for gi, x in enumerate(pilot):
        a, d0 = x.next_acceptor_pos, x.donor_pos; aD = a - 60      # slot_a = native pos a; slot_b = a-60
        P = x.sequence[a - 37:a + 3]                                # native partner window
        D, dg = decoys[int(rng.integers(0, len(decoys)))]
        while dg == x.gene_id:
            D, dg = decoys[int(rng.integers(0, len(decoys)))]
        ShA, ShB = sham40(P), sham40(D)                            # independent generic-AG shams per slot
        s3D = maxent.score3(D[17:40].upper(), matrix=M3) if set(D[17:40].upper()) <= set("ACGT") else float("nan")
        s3S = maxent.score3(ShB[17:40].upper(), matrix=M3) if set(ShB[17:40].upper()) <= set("ACGT") else float("nan")
        def build2(win_a, win_b, cA, cB):
            s = x.sequence[:aD - 37] + win_b + x.sequence[aD + 3:]  # slot_b (upstream), 40->40 in place
            s = s[:a - 37] + win_a + s[a + 3:]                      # slot_a (native pos), 40->40 in place
            if cA: s = substitute(s, a - 2, "CC")                  # corrupt slot_a AG
            if cB: s = substitute(s, aD - 2, "CC")                 # corrupt slot_b AG
            return s
        # native single-acceptor anchor (scaffold at slot_b, native P at slot_a)
        seqs.append(x.sequence); dons.append(d0); tags.append((gi, "nat"))
        seqs.append(substitute(x.sequence, a - 2, "CC")); dons.append(d0); tags.append((gi, "nat_c"))
        # RECIPROCAL SWAP: 3 configs x 3 cells (both-intact, slotA-corrupt, slotB-corrupt)
        #   N=native (P@a,D@b)  W=swap (D@a,P@b)  G=generic-AG (ShA@a,ShB@b, controls slot/composition/AG)
        for cn, (wa, wb) in {"N": (P, D), "W": (D, P), "G": (ShA, ShB)}.items():
            for cA, cB, sfx in [(0, 0, "00"), (1, 0, "A0"), (0, 1, "0B")]:
                seqs.append(build2(wa, wb, cA, cB)); dons.append(d0); tags.append((gi, cn + sfx))
        # OOD over BOTH slot windows: native scaffold vs config-N construct
        for lo, hi, nm in [(aD - 37, aD + 2, "b"), (a - 37, a + 2, "a")]:
            oseqs += [x.sequence, build2(P, D, 0, 0)]; owins += [(lo, hi), (lo, hi)]
            otags += [(gi, "ood_nat_" + nm), (gi, "ood_N_" + nm)]
        meta.append({"gene": x.gene_id, "s3_D": s3D, "s3_sham": s3S, "intron_len": a - d0})
    R = logits_at_donor(ad, task, seqs, dons, args.batch)
    by = {}
    for (gi, nm), r in zip(tags, R):
        by.setdefault(gi, {})[nm] = r
    ood = ood_nll(ad, task, oseqs, owins, args.batch)
    oby = {}
    for (gi, nm), v in zip(otags, ood):
        oby.setdefault(gi, {})[nm] = float(v)

    # ---- metrics (reciprocal swap: SLOT vs WINDOW ranking, net of generic-AG control) ----
    # reliance CAN be negative -> COMPETITION (intact rival depresses donor G-margin). Gates sign-agnostic.
    # relyA(cfg)=f(cfg,both) - f(cfg,slotA-corrupt); relyB(cfg)=f(cfg,both) - f(cfg,slotB-corrupt).
    THR = 0.05
    def relyA(b, c): return b[c + "00"][0] - b[c + "A0"][0]
    def relyB(b, c): return b[c + "00"][0] - b[c + "0B"][0]
    switch, fixed_agree = [], []
    slot_adv, slot_adv_G, win_adv = [], [], []                 # SLOT (position), generic-AG SLOT, WINDOW (seq)
    both_move, native_anchor, ood_delta = [], [], []
    rA_N, rB_N = [], []
    for gi in range(len(pilot)):
        b = by[gi]
        ams = [b["N" + s][1] for s in ("00", "A0", "0B")]
        switch.append(len(set(ams)) > 1)
        dA_sealed = relyA(b, "N")
        dA_fixed = [b["N00"][2][k] - b["NA0"][2][k] for k in range(3)]
        fixed_agree.append(all(np.sign(x) == np.sign(dA_sealed) for x in dA_fixed) if dA_sealed != 0 else True)
        aN, bN = relyA(b, "N"), relyB(b, "N")                  # native: P@a reliance, D@b reliance
        aW, bW = relyA(b, "W"), relyB(b, "W")                  # swap:   D@a reliance, P@b reliance
        aG, bG = relyA(b, "G"), relyB(b, "G")                  # generic AG at each slot
        rA_N.append(aN); rB_N.append(bN)
        both_move.append(abs(aN) > THR and abs(bN) > THR)
        # SLOT advantage (position a vs b), window-averaged over N,W:
        slot_adv.append(0.5 * ((aN - bN) + (aW - bW)))
        slot_adv_G.append(aG - bG)                             # positional prior reproducible by any AG
        # WINDOW advantage (P-seq vs D-seq), slot-averaged: Pwin=[aN,bW], Dwin=[bN,aW]:
        win_adv.append(0.5 * ((aN + bW) - (bN + aW)))
        native_anchor.append(b["nat"][0] - b["nat_c"][0])
        ood_delta.append(0.5 * ((oby[gi]["ood_N_a"] - oby[gi]["ood_nat_a"]) + (oby[gi]["ood_N_b"] - oby[gi]["ood_nat_b"])))
    sa, saG, wa = np.array(slot_adv), np.array(slot_adv_G), np.array(win_adv)
    slot_excess = sa - saG                                      # acceptor-SPECIFIC positional ranking beyond generic-AG
    zc = 2.8016
    rep = {
        "natural_yield": {"donors_with_2_acceptors": 142, "n_ge80nt_spacing": 0, "verdict": "INADEQUATE -> synthetic-only, constructed-setting scope"},
        "Q1_readout_stability": {"argmax_switch_rate": float(np.mean(switch)),
                                 "fixed_class_agreement_rate": float(np.mean(fixed_agree)),
                                 "PASS_90pct_stable_or_concordant": bool(np.mean([not s for s in switch]) >= 0.90 or np.mean(fixed_agree) >= 0.90)},
        "Q2_viability": {"both_measurably_move_frac": float(np.mean(both_move)),
                         "relyA_native_mean_Pslot": float(np.mean(rA_N)), "relyB_native_mean_Dslot": float(np.mean(rB_N)),
                         "native_anchor_mean": float(np.mean(native_anchor)),
                         "ood_delta_median": float(np.median(ood_delta)), "n": len(pilot)},
        "Q3_SLOT_vs_WINDOW_vs_AG": {
            "SLOT_adv_mean": float(np.mean(sa)), "SLOT_adv_sd": float(np.std(sa, ddof=1)),
            "SLOT_adv_generic_AG_mean": float(np.mean(saG)),
            "SLOT_excess_over_AG_mean": float(np.mean(slot_excess)), "SLOT_excess_sd": float(np.std(slot_excess, ddof=1)),
            "WINDOW_adv_mean": float(np.mean(wa)), "WINDOW_adv_sd": float(np.std(wa, ddof=1)),
            "criterion3_note": "PASS if SLOT_excess_over_AG or |WINDOW_adv| clears SESOI=0.05 (acceptor-SPECIFIC ranking, not generic-AG position)"},
        "MDE": {"expected_census_mde_at_N": {
            "SLOT_excess": {str(N): float(zc * np.std(slot_excess, ddof=1) / np.sqrt(N)) for N in (300, 800, 1500)},
            "WINDOW_adv": {str(N): float(zc * np.std(wa, ddof=1) / np.sqrt(N)) for N in (300, 800, 1500)}}},
        "score3_check": {"D_median": float(np.nanmedian([m["s3_D"] for m in meta])),
                         "sham_median": float(np.nanmedian([m["s3_sham"] for m in meta]))},
    }
    out = REPO / args.out; out.write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
