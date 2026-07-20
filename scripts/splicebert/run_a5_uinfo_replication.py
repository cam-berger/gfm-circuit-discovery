#!/usr/bin/env python
"""Donor-graph experiment 9 -- A5: independent replication of donor-side U_info.

Frozen protocol: studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_A5_UINFO_REPLICATION.md.

Validate the C16 U_info (top-k SVD of the clean-corrupt final donor residual) WITHOUT redefining it:
derive it on an independent gene split, compare subspaces (principal angles), test held-out
sufficiency/necessity, and run a rank curve vs assuming rank 8.  Uses the exact C16 definition.

Preflight/memorials FIRST (margins_fn_reproduces_cached licenses the readout); the run stops on
failure.  --smoke = memorials only (val sealed).
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

from gfmcd.provenance import ExperimentManifest, git_commit, sha256_file  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, gboot_ratio, load_adapter,
    load_examples, pick_device, ratio_of_means,
)
from gfmcd.splicing.subspace import (  # noqa: E402
    patch_component, random_subspace, subspace_overlap, top_k_subspace,
)
from gfmcd.splicing.unembed import margin_hard  # noqa: E402

RANKS = (1, 2, 4, 8, 16, 32)
SEED = 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n-null", type=int, default=50)
    ap.add_argument("--min-half", type=int, default=40)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="results/splicebert_coupling/a5_uinfo_replication.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    out = REPO / args.out
    pf_path = out.parent / "preflight_a5_uinfo_replication.json"
    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=True)
    task = CouplingTask(ad, batch=args.batch)
    off, bid, cls = task.off, task.bid, ad.model.cls
    dtype = next(cls.parameters()).dtype

    n = min(args.n, 40) if args.smoke else args.n
    X, _ = load_examples(args.gtf, args.fasta, args.split, n, task=task, tag=f"a5-{args.split}",
                         min_effect=None)
    if len(X) < 16:
        raise RuntimeError(f"only {len(X)} eligible examples.")
    strands = sorted({x.strand for x in X})

    def margins_fn(H_np):
        h = torch.from_numpy(np.ascontiguousarray(H_np, dtype=np.float32)).to(ad.device, dtype)
        return margin_hard(cls(h), bid).float().cpu().numpy()

    Hcl_L, Hco_L, mcl_L, mco_L, genes = [], [], [], [], []
    for i in range(0, len(X), args.batch):
        ch = X[i:i + args.batch]
        cl, co, dn = task.pair(ch)
        o_cl = ad.model(**cl, output_hidden_states=True)
        o_co = ad.model(**co, output_hidden_states=True)
        pos = torch.tensor([d + off for d in dn], device=dev)
        idx = torch.arange(len(dn), device=dev)
        Hcl_L.append(o_cl.hidden_states[-1][idx, pos].float().cpu().numpy())
        Hco_L.append(o_co.hidden_states[-1][idx, pos].float().cpu().numpy())
        mcl_L += task.margin(o_cl.logits.float(), dn).cpu().numpy().tolist()
        mco_L += task.margin(o_co.logits.float(), dn).cpu().numpy().tolist()
        genes += [x.gene_id for x in ch]
        print(f"  capture {min(i + args.batch, len(X))}/{len(X)}", end="\r", flush=True)
    print()
    Hcl = np.concatenate(Hcl_L); Hco = np.concatenate(Hco_L)
    mcl = np.array(mcl_L); mco = np.array(mco_L); coup = mcl - mco
    D = Hcl.shape[1]

    # deterministic disjoint gene split
    ug = sorted(set(genes))
    Agenes = set(ug[::2])
    inA = np.array([g in Agenes for g in genes])
    inB = ~inA
    coupling = float(np.mean(coup))
    mf_err = float(np.abs(margins_fn(Hcl) - mcl).max())

    checks = {
        "margins_fn_reproduces_cached": {"PASS": bool(mf_err < 1e-2), "max_abs_err": mf_err},
        "positive_control_partner_coupling": {"PASS": bool(coupling >= 0.20), "coupling": coupling},
        "both_genes_split": {"PASS": bool(inA.sum() >= args.min_half and inB.sum() >= args.min_half
                                          and len(Agenes & (set(genes) - Agenes)) == 0),
                             "nA": int(inA.sum()), "nB": int(inB.sum()),
                             "genes_A": len(Agenes), "genes_B": len(set(genes) - Agenes),
                             "disjoint": bool(len(Agenes & (set(genes) - Agenes)) == 0)},
        "both_strands_present": {"PASS": len(strands) == 2, "strands": strands},
    }
    ckpt = Path(args.model); ckpt = ckpt if ckpt.is_absolute() else REPO / ckpt
    checks["checkpoint_recorded"] = {"PASS": True, "sha256": sha256_file(ckpt / "pytorch_model.bin")}
    pf = {"commit": git_commit(REPO), "n": len(X), "ranks": list(RANKS), "checks": checks}
    if args.smoke:
        pf["checks"]["both_genes_split"]["PASS"] = True     # tiny-n split may be short; not gated in smoke
    pf["PASS"] = bool(all(c["PASS"] for c in pf["checks"].values()))
    pf_path.parent.mkdir(parents=True, exist_ok=True)
    pf_path.write_text(json.dumps(pf, indent=1))
    print(f"preflight PASS={pf['PASS']}  -> {pf_path}")
    for name, c in pf["checks"].items():
        print(f"    {'ok ' if c['PASS'] else 'XX '}{name}"
              + (f"  ({c.get('max_abs_err')})" if "max_abs_err" in c else ""))

    if args.smoke:
        print("SMOKE: preflight/memorials only; no rank curve reported (val sealed).")
        return
    if not pf["PASS"] and not args.force:
        print("  *** preflight FAILED -- not reporting the rank curve (prereg §3).")
        return

    rng = np.random.default_rng(SEED)
    dHA, dHB = (Hcl - Hco)[inA], (Hcl - Hco)[inB]

    def suff(U, mask):                                       # patch clean U-component into corrupt
        return ratio_of_means((margins_fn(patch_component(Hco[mask], Hcl[mask], U)) - mco[mask]).tolist(),
                              coup[mask].tolist())

    def nec(U, mask):                                        # remove clean U-component
        return ratio_of_means((mcl[mask] - margins_fn(patch_component(Hcl[mask], Hco[mask], U))).tolist(),
                              coup[mask].tolist())

    gA = [g for g, a in zip(genes, inA) if a]
    gB = [g for g, a in zip(genes, inA) if not a]
    rank_curve = {}
    for K in RANKS:
        UA, UB = top_k_subspace(dHA, K), top_k_subspace(dHB, K)
        ov = subspace_overlap(UA, UB)
        ov_null = float(np.mean([subspace_overlap(UA, random_subspace(D, K, rng)) for _ in range(args.n_null)]))
        # held-out sufficiency: A->B and B->A, averaged
        rAB, rBA = suff(UA, inB), suff(UB, inA)
        held = 0.5 * (rAB + rBA)
        insample = 0.5 * (suff(UB, inB) + suff(UA, inA))
        rnd = float(np.mean([0.5 * (suff(random_subspace(D, K, rng), inB)
                                    + suff(random_subspace(D, K, rng), inA)) for _ in range(args.n_null)]))
        necK = 0.5 * (nec(UA, inB) + nec(UB, inA))
        entry = {"overlap": ov, "overlap_null": ov_null, "R_suff_heldout": held,
                 "R_suff_insample": insample, "R_suff_random": rnd, "N_necessity_heldout": necK}
        if K == 8:
            entry["R_suff_heldout_AB_ci"] = list(gboot_ratio(
                (margins_fn(patch_component(Hco[inB], Hcl[inB], UA)) - mco[inB]).tolist(),
                coup[inB].tolist(), gB))
        rank_curve[K] = entry
        print(f"  K={K:2d}  overlap {ov:.3f} (null {ov_null:.3f})  R_suff held {held:+.3f} "
              f"insample {insample:+.3f} rand {rnd:+.3f}  N {necK:+.3f}")

    r8 = rank_curve[8]
    result = {
        "n": len(X), "n_genes": len(set(genes)), "strands": strands, "coupling": coupling,
        "nA": int(inA.sum()), "nB": int(inB.sum()), "ranks": list(RANKS),
        "rank_curve": {str(k): v for k, v in rank_curve.items()},
        "headline_rank8": r8, "preflight_PASS": pf["PASS"],
        "meets_power_target": bool(len(X) >= 250 and inA.sum() >= 40 and inB.sum() >= 40),
        "confirmatory": bool(pf["PASS"] and len(X) >= 250),
        "REFLECTION": {"status": "PENDING -- fill after inspection (prereg §5)"},
    }
    out.write_text(json.dumps(result, indent=1))
    status = "CONFIRMATORY" if result["confirmatory"] else "DIAGNOSTIC"
    man = ExperimentManifest.create(
        "splicebert_coupling:a5_uinfo_replication", sys.argv, status=status,
        config={"split": args.split, "n": len(X), "n_genes": result["n_genes"], "ranks": list(RANKS),
                "n_null": args.n_null, "preflight_PASS": pf["PASS"],
                "meets_power_target": result["meets_power_target"]},
        seeds={"gene_split": 0, "random_subspaces": SEED},
        notes=["Independent replication of donor-side U_info (C16): disjoint gene split, subspace "
               "overlap vs random, held-out sufficiency/necessity, rank curve. U_info NOT redefined.",
               f"status={status} (preflight_PASS={pf['PASS']}, power={result['meets_power_target']})."])
    man.add_input(str(ckpt / "pytorch_model.bin"))
    for data in (args.gtf, args.fasta):
        if Path(data).exists():
            man.add_input(data)
    man.write(out.parent / (out.stem + ".manifest.json"))

    print("=" * 80)
    print(f"  rank-8: overlap {r8['overlap']:.3f} (null {r8['overlap_null']:.3f})  "
          f"held-out R_suff {r8['R_suff_heldout']:+.3f} vs in-sample {r8['R_suff_insample']:+.3f} "
          f"vs random {r8['R_suff_random']:+.3f}  necessity {r8['N_necessity_heldout']:+.3f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
