#!/usr/bin/env python
"""THE FORWARD ARM -- does L3H1 deliver U_info?  (donor-graph 1b)

1a anchored the OUTPUT: the coupling lives in a low-rank object U_info (top-SVD of the
clean-corrupt final residual), which recovers 0.92 of the coupling (C16).  readout_path.json
showed restoring L3H1's clean output into the corrupt run recovers most of the margin through a
distributed downstream chain.  This joins them:

    is the margin L3H1 restores CARRIED BY U_info, or delivered through directions the anchor
    never saw?

Restore L3H1 (patch_heads, downstream free), then ROUTE the restored final residual through
U_info: patch only U_info's component of h_re into h_co and read the margin.  If that recovers
as much as the full restoration, L3H1's effect flows through U_info and the forward arm meets
the backward arm.  The near-tautology (h_re ~ h_cl, whose U_info-component is clean by
construction) is defused by (a) partial restoration (~0.82) and (b) a size-matched random
rank-8 null -- the control that can fail.

Prereg (FROZEN): studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_DONOR_FORWARD.md
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
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, READER, CouplingTask, capture_head_outputs,
    capture_writes, downstream_blocks, gboot, gboot_ratio, load_adapter, load_examples,
    patch_heads, pick_device, ratio_of_means,
)
from gfmcd.splicing.subspace import (  # noqa: E402
    patch_component, project, random_subspace, subspace_overlap, top_k_subspace,
)
from gfmcd.splicing.unembed import margin_hard  # noqa: E402

K = 8                     # U_info rank (the anchor's 0.92-recovery rank)
N_NULL = 200
SEED = 0
BLOCK_LABEL = {(3, "mlp"): "MLP3", (4, "attn"): "L4-attn", (4, "mlp"): "MLP4",
               (5, "attn"): "L5-attn", (5, "mlp"): "MLP5"}


def donor_rows(o, off, dn, dtype):
    """(final donor residual [B,hidden] np, margin-input logits at donor) for a forward output."""
    pos = torch.tensor([d + off for d in dn], device=o.logits.device)
    idx = torch.arange(len(dn), device=o.logits.device)
    H = o.hidden_states[-1][idx, pos].float().cpu().numpy()
    return H, pos, idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="results/splicebert_coupling/donor_forward.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)          # inference only; capture forwards must not build graphs

    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=True)      # patch_heads on the proven template
    task = CouplingTask(ad, batch=args.batch)
    off, bid, cls = task.off, task.bid, ad.model.cls
    dtype = next(cls.parameters()).dtype
    DOWN = downstream_blocks(ad, READER[0])             # [(3,mlp),(4,attn),(4,mlp),(5,attn),(5,mlp)]

    X, _ = load_examples(args.gtf, args.fasta, args.split, args.n, task=task,
                         tag=f"forward-{args.split}", min_effect=None)
    genes = [x.gene_id for x in X]
    print(f"{len(X)} unconditional {args.split} exons   device={dev}   DOWN={[BLOCK_LABEL[b] for b in DOWN]}")

    H_cl, H_co, H_re, H_rf, H_ro = [], [], [], [], []   # clean, corrupt, restored all / flank / elsewhere
    m_cl, m_co, m_re, m_rf, m_ro = [], [], [], [], []
    W_cl = {b: [] for b in DOWN}                        # block writes at donor, per condition
    W_co = {b: [] for b in DOWN}
    W_re = {b: [] for b in DOWN}

    def gather(o, dn):
        H, _, _ = donor_rows(o, off, dn, dtype)
        m = task.margin(o.logits.float(), dn).cpu().numpy()
        return H, m

    def block_writes_at_donor(store, dn):
        device = store[DOWN[0]].device
        idx = torch.arange(len(dn), device=device)
        pos = torch.tensor([d + off for d in dn], device=device)
        return {b: store[b][idx, pos].float().cpu().numpy() for b in DOWN}

    for i in range(0, len(X), args.batch):
        ch = X[i:i + args.batch]
        cl, co, dn = task.pair(ch)
        S = cl["input_ids"].shape[1]

        # --- clean: capture head outputs (for restoration) + block writes + hidden, one forward
        tmp = {}
        wcl = capture_writes(ad, DOWN, lambda: tmp.__setitem__(
            "heads", capture_head_outputs(ad, lambda: tmp.__setitem__(
                "o", ad.model(**cl, output_hidden_states=True)))))
        clean_heads, ocl = tmp["heads"], tmp["o"]

        # --- corrupt: block writes + hidden
        tmp2 = {}
        wco = capture_writes(ad, DOWN, lambda: tmp2.__setitem__(
            "o", ad.model(**co, output_hidden_states=True)))
        oco = tmp2["o"]

        # --- restored (all positions): patch L3H1 clean output into corrupt, downstream live
        tmp3 = {}
        with patch_heads(ad, clean_heads, [READER]):
            wre = capture_writes(ad, DOWN, lambda: tmp3.__setitem__(
                "o", ad.model(**co, output_hidden_states=True)))
        ore = tmp3["o"]

        # --- restored (donor +/-5 flank only): token-masked restoration
        flank = torch.zeros(len(ch), S, dtype=torch.bool, device=ad.device)
        for j, x in enumerate(ch):
            for k in range(-5, 6):
                p = x.donor_pos + off + k
                if 0 <= p < S:
                    flank[j, p] = True
        with patch_heads(ad, clean_heads, [READER], token_mask=flank):
            orf = ad.model(**co, output_hidden_states=True)

        # --- restored (everywhere EXCEPT donor +/-5): the localization control
        other = co["attention_mask"].bool() & ~flank
        with patch_heads(ad, clean_heads, [READER], token_mask=other):
            oro = ad.model(**co, output_hidden_states=True)

        for cond, o, Hl, ml in ((0, ocl, H_cl, m_cl), (1, oco, H_co, m_co),
                                (2, ore, H_re, m_re), (3, orf, H_rf, m_rf),
                                (4, oro, H_ro, m_ro)):
            H, m = gather(o, dn)
            Hl.append(H); ml.append(m)
        for b in DOWN:
            W_cl[b].append(block_writes_at_donor(wcl, dn)[b])
            W_co[b].append(block_writes_at_donor(wco, dn)[b])
            W_re[b].append(block_writes_at_donor(wre, dn)[b])
        print(f"  fwd {min(i + args.batch, len(X))}/{len(X)}", end="\r", flush=True)
    print()

    cat = lambda L: np.concatenate(L)
    H_cl, H_co, H_re, H_rf, H_ro = map(cat, (H_cl, H_co, H_re, H_rf, H_ro))
    m_cl, m_co, m_re, m_rf, m_ro = map(cat, (m_cl, m_co, m_re, m_rf, m_ro))
    coup = m_cl - m_co
    coup_lo, coup_hi = gboot(coup, genes)

    U_info = top_k_subspace(H_cl - H_co, K)

    def margins_fn(H_np):
        h = torch.from_numpy(np.ascontiguousarray(H_np, dtype=np.float32)).to(ad.device, dtype)
        with torch.no_grad():
            return margin_hard(cls(h), bid).float().cpu().numpy()

    mf_check = float(np.abs(margins_fn(H_cl) - m_cl).max())    # sanity: cls reproduces cached margins
    if mf_check > 1e-2:
        sys.exit(f"margins_fn(H_cl) does not reproduce cached margins (max err {mf_check:.2e}); "
                 "the logit-lens or dtype is inconsistent -- every routed recovery would be wrong. Stop.")

    # ---- gates -------------------------------------------------------------------------------
    R_L3 = ratio_of_means(m_re - m_co, coup)
    R_L3_ci = list(gboot_ratio(m_re - m_co, coup, genes))
    m_ru = margins_fn(patch_component(H_co, H_re, U_info))
    R_U = ratio_of_means(m_ru - m_co, coup)
    R_U_ci = list(gboot_ratio(m_ru - m_co, coup, genes))

    rng = np.random.default_rng(SEED)
    D = H_cl.shape[1]
    nulls = np.array([
        ratio_of_means(margins_fn(patch_component(H_co, H_re, random_subspace(D, K, rng))) - m_co, coup)
        for _ in range(N_NULL)])
    p_route = empirical_upper_p_value(R_U, nulls)

    m_rf_u = margins_fn(patch_component(H_co, H_rf, U_info))
    R_U_flank = ratio_of_means(m_rf_u - m_co, coup)
    R_L3_flank = ratio_of_means(m_rf - m_co, coup)
    m_ro_u = margins_fn(patch_component(H_co, H_ro, U_info))
    R_U_other = ratio_of_means(m_ro_u - m_co, coup)         # restore everywhere EXCEPT donor +/-5

    # F-Align (descriptive)
    U_re = top_k_subspace(H_re - H_co, K)
    align = subspace_overlap(U_re, U_info)
    align_rand = float(np.mean([subspace_overlap(random_subspace(D, K, rng),
                                                 random_subspace(D, K, rng)) for _ in range(50)]))

    # per-block U_info-write magnitude (descriptive, approximate: ignores inter-layer LN)
    assembly = {}
    for b in DOWN:
        wcl_b, wco_b, wre_b = cat(W_cl[b]), cat(W_co[b]), cat(W_re[b])
        norm = lambda W: float(np.mean(np.linalg.norm(project(W, U_info), axis=1)))
        assembly[BLOCK_LABEL[b]] = {"clean": norm(wcl_b), "corrupt": norm(wco_b),
                                    "restored": norm(wre_b)}

    verdict = {
        "precondition_R_L3_ge_0.5": R_L3 >= 0.50,
        "F_route": (R_U / max(R_L3, 1e-9)) >= 0.80 and R_U > float(nulls.max()) and p_route <= 0.05,
        "F_local": R_U_flank >= 0.50 * R_U,
    }
    verdict["overall"] = all(verdict.values())

    R = {
        "split": args.split, "n": len(X), "device": dev, "K": K, "margins_fn_max_err": mf_check,
        "behavior": {"coupling_mean": float(coup.mean()), "coupling_ci": [coup_lo, coup_hi],
                     "m_clean": float(m_cl.mean()), "m_corrupt": float(m_co.mean()),
                     "m_restored": float(m_re.mean())},
        "F_route": {"R_L3": R_L3, "R_L3_ci": R_L3_ci, "R_U": R_U, "R_U_ci": R_U_ci,
                    "routed_fraction": float(R_U / max(R_L3, 1e-9)),
                    "null_max": float(nulls.max()), "null_mean": float(nulls.mean()),
                    "p_add_one": p_route},
        "F_local": {"R_U_all": R_U, "R_U_flank": R_U_flank, "R_U_other": R_U_other,
                    "R_L3_flank": R_L3_flank,
                    "flank_fraction": float(R_U_flank / max(R_U, 1e-9)),
                    "other_fraction": float(R_U_other / max(R_U, 1e-9))},
        "F_align": {"overlap": align, "random": align_rand},
        "assembly_uinfo_write_norm": assembly,
        "verdict": {k: bool(v) for k, v in verdict.items()},
        "REFLECTION": {"status": "PENDING -- fill after inspection (prereg section 5)"},
    }

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=1))
    ExperimentManifest.create(
        "splicebert_coupling:donor_forward", sys.argv,
        config={"split": args.split, "n": len(X), "reader": list(READER), "K": K,
                "n_null": N_NULL, "seed": SEED, "downstream_blocks": [list(b) for b in DOWN]},
        notes=["Forward arm (1b): attributes the anchor's U_info to L3H1's write.",
               "R_U routes the restored final residual through U_info; null = random rank-8.",
               "Per-block assembly is an approximate U_info-projection (ignores inter-layer LN)."],
    ).write(out.parent / "donor_forward.manifest.json")

    print("\n" + "=" * 78)
    print(f"coupling {coup.mean():+.3f} [{coup_lo:+.3f},{coup_hi:+.3f}]   "
          f"restore-L3H1 margin {m_re.mean():+.3f}")
    print(f"F-Route:  R_L3 {R_L3:.3f} {tuple(round(x,3) for x in R_L3_ci)}   "
          f"R_U {R_U:.3f} {tuple(round(x,3) for x in R_U_ci)}   "
          f"routed {100 * R_U / max(R_L3,1e-9):.0f}%   null_max {nulls.max():.3f}   p {p_route:.4f}")
    print(f"F-Local:  R_U(all) {R_U:.3f}   R_U(donor+/-5) {R_U_flank:.3f} "
          f"({100 * R_U_flank / max(R_U,1e-9):.0f}%)   R_U(elsewhere) {R_U_other:.3f} "
          f"({100 * R_U_other / max(R_U,1e-9):.0f}%)")
    print(f"F-Align:  overlap(top-SVD(h_re-h_co), U_info) {align:.3f}  (random {align_rand:.3f})")
    print("U_info-write magnitude per block (clean / corrupt / restored):")
    for b in DOWN:
        a = assembly[BLOCK_LABEL[b]]
        print(f"    {BLOCK_LABEL[b]:8s} {a['clean']:.3f} / {a['corrupt']:.3f} / {a['restored']:.3f}")
    print("\nVERDICT:")
    for k, v in verdict.items():
        print(f"  {k:26s} {'PASS' if v else 'FAIL'}")
    print("=" * 78)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
