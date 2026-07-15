#!/usr/bin/env python
"""ANCHOR THE DONOR LOGIT -- the backward foundation of the L3H1->donor-logit trace.

The whole meet-in-the-middle graph explains one scalar: the model's preference for G at the
masked donor (`margin = logit(G) - max logit(A,C,T)`).  Before tracing anything backward from
the output we must know (a) that `cls(final_residual)` reproduces that logit exactly [G0], and
(b) which low-dimensional residual direction produces the CLEAN-vs-CORRUPT coupling.

Approach 3: characterise BOTH the local read direction `U_read` (gradient of the margin) and
the information object `U_info` (top-SVD of clean-corrupt), and test their alignment.  The SVD
cannot fail, so the decomposition is licensed only by controls that can: a size-matched
random-subspace null (C-Null) and a destructive alive-check (C-Alive, the R8 lesson).

Every causal test runs on cached final residuals via `cls()` -- no re-forward.

Prereg (FROZEN): studies/splicebert_partner_selection/PREREGISTRATION_DONOR_ANCHOR.md
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.rigor import empirical_upper_p_value  # noqa: E402
from gfmcd.splicing.anchor_gates import (  # noqa: E402
    alive_margin, nec_fraction, null_recoveries, suff_recovery,
)
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, gboot, gboot_ratio,
    load_adapter, load_examples, pick_device, ratio_of_means,
)
from gfmcd.splicing.subspace import (  # noqa: E402
    energy_spectrum, patch_component, random_subspace, subspace_overlap, top_k_subspace,
)
from gfmcd.splicing.unembed import margin_hard, read_gradient  # noqa: E402

KS = (1, 2, 4, 8)                 # ranks characterised
GATE_KS = (1, 8)                  # ranks with frozen pass/fail thresholds
N_NULL = 200                      # size-matched random subspaces
SEED = 0


def pick_nonsite_g(x, min_gap: int = 30) -> int | None:
    """A masked-G control site >= min_gap nt from both the donor and its partner acceptor."""
    d, a = x.donor_pos, x.next_acceptor_pos
    cands = [p for p in range(len(x.sequence))
             if x.sequence[p] == "G" and abs(p - d) >= min_gap
             and (a is None or abs(p - a) >= min_gap)]
    if not cands:
        return None
    mid = len(x.sequence) // 2
    return min(cands, key=lambda p: abs(p - mid))     # deterministic: nearest to centre


def gather_residuals(ad, task, off, positions_enc, positions_idx):
    """(final residual [B,hidden], margin [B]) at `positions_idx` for an encoded batch."""
    with torch.no_grad():
        o = ad.model(**positions_enc, output_hidden_states=True)
    pos = torch.tensor([p + off for p in positions_idx], device=ad.device)
    idx = torch.arange(len(positions_idx), device=ad.device)
    H = o.hidden_states[-1][idx, pos].float().cpu().numpy()
    m = task.margin(o.logits.float(), positions_idx).cpu().numpy()
    return H, m, o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="results/splicebert_coupling/donor_anchor.json")
    args = ap.parse_args()

    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=False)     # attention is NOT read here
    task = CouplingTask(ad, batch=args.batch)
    off, bid, cls = task.off, task.bid, ad.model.cls
    dtype = next(cls.parameters()).dtype

    # UNCONDITIONAL set (min_effect=None): no conditioning on coupling (dodges R10).
    X, _ = load_examples(args.gtf, args.fasta, args.split, args.n, task=task,
                         tag=f"anchor-{args.split}", min_effect=None)
    genes = [x.gene_id for x in X]
    print(f"{len(X)} unconditional {args.split} exons   device={dev}")

    # ---- cache clean / corrupt final residuals + margins, and check G0 on batch 0 -----------
    H_cl, H_co, m_cl, m_co = [], [], [], []
    g0 = None
    for i in range(0, len(X), args.batch):
        ch = X[i:i + args.batch]
        cl, co, dn = task.pair(ch)
        hc, mc, ocl = gather_residuals(ad, task, off, cl, dn)
        hk, mk, _ = gather_residuals(ad, task, off, co, dn)
        H_cl.append(hc); H_co.append(hk); m_cl.append(mc); m_co.append(mk)
        if g0 is None:                                   # G0: cls(final residual) == logits
            pos = torch.tensor([d + off for d in dn], device=ad.device)
            idx = torch.arange(len(ch), device=ad.device)
            with torch.no_grad():
                relog = cls(ocl.hidden_states[-1].to(dtype))[idx, pos].float()
            g0 = float((relog - ocl.logits.float()[idx, pos]).abs().max().item())
        print(f"  cache {min(i + args.batch, len(X))}/{len(X)}", end="\r", flush=True)
    print()
    H_cl = np.concatenate(H_cl); H_co = np.concatenate(H_co)
    m_cl = np.concatenate(m_cl); m_co = np.concatenate(m_co)

    if g0 >= 1e-3:
        sys.exit(f"G0 FAILED: max|cls(hidden[-1]) - logits| = {g0:.2e} >= 1e-3. "
                 "The unembed does not reproduce the native logits; stop.")
    print(f"G0 OK   max|cls(hidden[-1]) - logits| = {g0:.2e}")

    # ---- the no-info floor and baseline residual (masked non-site G) -------------------------
    ns = [(x, p) for x, p in ((x, pick_nonsite_g(x)) for x in X) if p is not None]
    H_ns, m_ns = [], []
    for i in range(0, len(ns), args.batch):
        chunk = ns[i:i + args.batch]
        seqs = [x.sequence for x, _ in chunk]
        ps = [p for _, p in chunk]
        enc = task.encode(seqs, ps)
        h, m, _ = gather_residuals(ad, task, off, enc, ps)
        H_ns.append(h); m_ns.append(m)
    H_ns = np.concatenate(H_ns); m_ns = np.concatenate(m_ns)
    floor = float(m_ns.mean())
    h_baseline = H_ns.mean(0)

    coup = m_cl - m_co
    coup_lo, coup_hi = gboot(coup, genes)
    print(f"coupling {coup.mean():+.3f} [{coup_lo:+.3f},{coup_hi:+.3f}]   "
          f"clean margin {m_cl.mean():+.3f}   floor(non-site G) {floor:+.3f}   "
          f"clean-floor {m_cl.mean() - floor:+.3f}   (n_floor={len(m_ns)})")

    # ---- subspaces: U_info (clean-corrupt) and U_read (gradient) -----------------------------
    delta = H_cl - H_co
    G = read_gradient(cls, torch.from_numpy(H_cl).to(ad.device, dtype), bid).float().cpu().numpy()
    U_info = {k: top_k_subspace(delta, k) for k in KS}
    U_read = {k: top_k_subspace(G, k) for k in KS}

    def margins_fn(H_np):
        h = torch.from_numpy(np.ascontiguousarray(H_np, dtype=np.float32)).to(ad.device, dtype)
        with torch.no_grad():
            return margin_hard(cls(h), bid).float().cpu().numpy()

    # sanity: margins_fn(H_cl) must reproduce the cached m_cl
    mf_check = float(np.abs(margins_fn(H_cl) - m_cl).max())

    # ---- D1 rank spectrum -------------------------------------------------------------------
    sp_d, sp_g = energy_spectrum(delta), energy_spectrum(G)
    def dims_to(frac, sp): return int(np.searchsorted(sp, frac) + 1)
    D1 = {"delta_dims_80": dims_to(0.80, sp_d), "delta_dims_90": dims_to(0.90, sp_d),
          "delta_top1_energy": float(sp_d[0]), "delta_top8_energy": float(sp_d[min(7, len(sp_d) - 1)]),
          "grad_dims_80": dims_to(0.80, sp_g), "grad_dims_90": dims_to(0.90, sp_g)}

    # ---- D2 alignment (read vs info) vs random-overlap baseline ------------------------------
    rng = np.random.default_rng(SEED)
    D = H_cl.shape[1]
    D2 = {}
    for k in KS:
        rand = float(np.mean([subspace_overlap(random_subspace(D, k, rng),
                                               random_subspace(D, k, rng)) for _ in range(50)]))
        D2[f"k{k}"] = {"overlap": subspace_overlap(U_read[k], U_info[k]), "random": rand}

    # ---- D3 linearity: local-linear prediction of the coupling -------------------------------
    lin = np.sum(G * delta, axis=1)                      # g_i . Delta_h_i
    D3 = {"corr": float(np.corrcoef(lin, coup)[0, 1]),
          "frac_explained": ratio_of_means(lin, coup)}

    # ---- causal gates: C-Suff, C-Nec, C-Null (info + read, k in {1,8}) -----------------------
    gates: dict = {"info": {}, "read": {}}
    for name, Uset in (("info", U_info), ("read", U_read)):
        for k in GATE_KS:
            U = Uset[k]
            m_suff = margins_fn(patch_component(H_co, H_cl, U))
            m_nec = margins_fn(patch_component(H_cl, H_co, U))
            suff = ratio_of_means(m_suff - m_co, coup)
            nec = ratio_of_means(m_cl - m_nec, coup)
            nulls = null_recoveries(margins_fn, H_cl, H_co, m_cl, m_co, k, N_NULL,
                                    np.random.default_rng(SEED))
            gates[name][f"k{k}"] = {
                "suff": suff, "suff_ci": list(gboot_ratio(m_suff - m_co, coup, genes)),
                "nec": nec, "nec_ci": list(gboot_ratio(m_cl - m_nec, coup, genes)),
                "null_max": float(nulls.max()), "null_mean": float(nulls.mean()),
                "p_add_one": empirical_upper_p_value(suff, nulls),
            }

    # ---- C-Alive (destructive alive-check, R8) ----------------------------------------------
    alive = alive_margin(margins_fn, h_baseline, len(H_cl))

    # ---- verdicts vs FROZEN thresholds ------------------------------------------------------
    gi = gates["info"]
    verdict = {
        "G0": g0 < 1e-3,
        "C_suff": gi["k1"]["suff"] >= 0.50 and gi["k8"]["suff"] >= 0.80,
        "C_nec": gi["k8"]["nec"] >= 0.80,
        "C_null": all(gi[f"k{k}"]["suff"] > gi[f"k{k}"]["null_max"]
                      and gi[f"k{k}"]["p_add_one"] <= 0.05 for k in GATE_KS),
        "C_alive": alive <= floor + 0.10,
    }
    verdict["overall"] = all(verdict.values())

    R = {
        "split": args.split, "n": len(X), "n_floor": len(m_ns), "device": dev,
        "G0_max_abs": g0, "margins_fn_max_err": mf_check,
        "behavior": {"m_clean_mean": float(m_cl.mean()), "m_corrupt_mean": float(m_co.mean()),
                     "coupling_mean": float(coup.mean()), "coupling_ci": [coup_lo, coup_hi],
                     "floor": floor, "clean_minus_floor": float(m_cl.mean() - floor)},
        "D1_rank_spectrum": D1, "D2_alignment": D2, "D3_linearity": D3,
        "gates": gates, "C_alive": {"alive_margin": alive, "floor": floor,
                                    "threshold": floor + 0.10, "pass": bool(verdict["C_alive"])},
        "verdict": {k: bool(v) for k, v in verdict.items()},
        "REFLECTION": {"status": "PENDING -- fill after inspection (prereg section 7)"},
    }

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=1))
    ExperimentManifest.create(
        "splicebert_coupling:donor_anchor", sys.argv,
        config={"split": args.split, "n": len(X), "ranks": list(KS), "n_null": N_NULL, "seed": SEED},
        notes=["Anchor (donor-graph 1a): backward foundation of the L3H1->donor-logit trace.",
               "All causal tests run on cached final residuals via cls() -- no re-forward.",
               "C-Nec is the dual of C-Suff (prereg amendment 2026-07-14).",
               "The SVD cannot fail; C-Null and C-Alive carry the falsifiable weight."],
    ).write(out.parent / "donor_anchor.manifest.json")

    # ---- report -----------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"D1  clean-corrupt rank: {D1['delta_dims_80']} dims to 80%, "
          f"{D1['delta_dims_90']} to 90%   (top1={D1['delta_top1_energy']:.2f}, "
          f"top8={D1['delta_top8_energy']:.2f})")
    print("D2  read vs info subspace overlap (vs random):")
    for k in KS:
        print(f"      k={k}: {D2[f'k{k}']['overlap']:.3f}  (random {D2[f'k{k}']['random']:.3f})")
    print(f"D3  local-linear explains {100 * D3['frac_explained']:.0f}% of coupling "
          f"(corr {D3['corr']:.2f})")
    print("\nCAUSAL GATES (info subspace):")
    for k in GATE_KS:
        g = gi[f"k{k}"]
        print(f"  k={k}: suff {g['suff']:.3f} {tuple(round(x,3) for x in g['suff_ci'])}  "
              f"nec {g['nec']:.3f}  null_max {g['null_max']:.3f}  p {g['p_add_one']:.4f}")
    print(f"C-Alive: inject baseline -> margin {alive:+.3f}  (floor {floor:+.3f}, "
          f"threshold {floor + 0.10:+.3f})")
    print("\nVERDICT (vs frozen thresholds):")
    for k, v in verdict.items():
        print(f"  {k:9s} {'PASS' if v else 'FAIL'}")
    print("=" * 78)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
