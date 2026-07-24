#!/usr/bin/env python
"""Donor-graph experiment 8 -- A4: is L2H15's address contribution absolute or relative position?

Frozen protocol: studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_A4_ADDRESS_GEOMETRY.md.

L2H15's contribution to targeting G15 = S_A^intact - S_A^{L2H15-ablated}, where S_A = L3H1's
donor-query QK-logit PREFERENCE for the intron-mate acceptor a+ over a phase-matched decoy a-
(the A1a targeting score, where L2H15's contribution robustly lives -- C20).  Intervene ON POSITION
EMBEDDINGS ONLY.  Two operations:
  1. common translation: donor + a+ + a- all shifted by D (every offset preserved) -> E_common.
  2. differential: a+ window only shifted by D (d->a+ offset changes) -> E_diff.

Preflight/memorials FIRST; the run stops on failure.  --smoke = memorials only (val sealed).
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest, git_commit, sha256_file  # noqa: E402
from gfmcd.splicing.address import phase_matched_decoy, qk_logit_to_window, windows  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    ADDRESSER, DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, READER, CouplingTask, ablate,
    capture_proj, gboot, load_adapter, load_examples, pick_device,
)
from gfmcd.splicing.provenance import clamp_embed_component, embed_component  # noqa: E402

L3, H3 = READER
OFFS = (-3, -2, -1, 0)
GRID = (-96, -48, -24, 0, 24, 48, 96)
SPLIT_CHROMS = {"val": ("chr11", "chr16", "chr19"), "test": ("chr8", "chr9")}


def tracked_tokens(pairs, off, device):
    donor = torch.tensor([[x.donor_pos + off] for x, _ in pairs], device=device)   # [B,1]
    aplus = windows([x.next_acceptor_pos for x, _ in pairs], off, device, offsets=OFFS)
    aminus = windows([am for _, am in pairs], off, device, offsets=OFFS)
    return torch.cat([donor, aplus, aminus], dim=1), donor[:, 0], aplus, aminus     # [B,9]


def shift_source(PE, tracked, shifts, S):
    """[B,S,H] PE source: token ``tracked[b,j]`` -> ``PE[tracked[b,j]+shifts[j]]``. shifts len 9."""
    sh = torch.tensor(shifts, device=tracked.device)
    idx = (tracked + sh).clamp(0, PE.shape[0] - 1)
    src = torch.zeros(tracked.shape[0], S, PE.shape[1], device=tracked.device, dtype=PE.dtype)
    b = torch.arange(tracked.shape[0], device=tracked.device)[:, None].expand_as(tracked)
    src[b, tracked] = PE[idx]
    return src


def s_a(ad, enc, donor_tok, aplus, aminus, Dh, *, source=None, tracked=None, ablate_head=None):
    """L3H1 donor-query QK-logit preference ell(d->a+) - ell(d->a-), optionally under a PE clamp
    and/or L2H15 ablation."""
    with ExitStack() as st:
        if ablate_head is not None:
            st.enter_context(ablate(ad, [ablate_head])())
        if source is not None:
            st.enter_context(clamp_embed_component(ad, "position", source, tracked))
        proj = capture_proj(ad, L3, ["query", "key"], lambda: ad.model(**enc))
    q, k = proj["query"], proj["key"]
    return (qk_logit_to_window(q, k, H3, Dh, donor_tok, aplus)
            - qk_logit_to_window(q, k, H3, Dh, donor_tok, aminus))


def toeplitz_fraction(PE, Wq, Wk, Dh, lo=100, hi=900, step=8):
    idx = torch.arange(lo, hi, step, device=PE.device)
    K = ((PE[idx] @ Wq.T) @ (PE[idx] @ Wk.T).T) / (Dh ** 0.5)
    G = K.shape[0]
    resid = [torch.diagonal(K, offset=d) - torch.diagonal(K, offset=d).mean()
             for d in range(-(G - 1), G) if torch.diagonal(K, offset=d).numel() >= 3]
    within = float((torch.cat(resid) ** 2).mean())
    total = float(((K - K.mean()) ** 2).mean())
    return within / total if total else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--min-n", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tol", type=int, default=40)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="results/splicebert_coupling/superseded/a4_address_geometry.json")
    args = ap.parse_args()

    out = REPO / args.out
    pf_path = out.parent / "preflight_a4_address_geometry.json"
    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=True)
    task = CouplingTask(ad, batch=args.batch)
    off = task.off
    Dh = ad.hidden_size // ad.num_heads
    PE = embed_component(ad, "position").weight.detach()
    P = PE.shape[0]
    Wq = ad.attention_self_modules[L3].query.weight[H3 * Dh:(H3 + 1) * Dh, :].detach()
    Wk = ad.attention_self_modules[L3].key.weight[H3 * Dh:(H3 + 1) * Dh, :].detach()

    Xpool, _ = load_examples(args.gtf, args.fasta, args.split, args.n, task=task,
                             tag=f"a4-{args.split}", min_effect=None)
    dmin, dmax = min(GRID), max(GRID)

    def fits(x, am):
        toks = ([x.donor_pos + off]
                + [x.next_acceptor_pos + off + k for k in OFFS]
                + [am + off + k for k in OFFS])
        return (min(toks) + dmin >= 0) and (max(toks) + dmax <= P - 1)
    pairs = []
    for x in Xpool:
        am = phase_matched_decoy(x, tol=args.tol)
        if am is not None and fits(x, am):
            pairs.append((x, am))
    if args.smoke:
        pairs = pairs[:24]
    if len(pairs) < 8:
        raise RuntimeError(f"only {len(pairs)} decoy+grid-fitting pairs of {len(Xpool)}.")
    strands = sorted({x.strand for x, _ in pairs})
    genes = [x.gene_id for x, _ in pairs]
    print(f"{len(pairs)} decoy+grid-fitting pairs (of {len(Xpool)}); strands={strands}; P={P}; "
          f"device={dev}{'  [SMOKE]' if args.smoke else ''}")

    Gc = {d: [] for d in GRID}                              # common
    Gd = {d: [] for d in GRID}                              # differential
    Sc_int = {d: [] for d in GRID}                          # diagnostic: intact S_A under common
    Sc_abl = {d: [] for d in GRID}                          # diagnostic: ablated S_A under common
    shift0_err = 0.0; prob_inv = None; qk_err = 0.0
    with torch.no_grad():
        for i in range(0, len(pairs), args.batch):
            ch = pairs[i:i + args.batch]
            exs = [x for x, _ in ch]
            dn = [x.donor_pos for x in exs]
            enc = task.encode([x.sequence for x in exs], dn)
            S = enc["input_ids"].shape[1]
            tracked, donor_tok, aplus, aminus = tracked_tokens(ch, off, ad.device)
            for d in GRID:
                sc = shift_source(PE, tracked, [d] * 9, S)                     # common
                si = s_a(ad, enc, donor_tok, aplus, aminus, Dh, source=sc, tracked=tracked)
                sa = s_a(ad, enc, donor_tok, aplus, aminus, Dh, source=sc, tracked=tracked,
                         ablate_head=ADDRESSER)
                gc = si - sa
                Sc_int[d] += si.cpu().tolist(); Sc_abl[d] += sa.cpu().tolist()
                Gc[d] += gc.cpu().tolist()
                if d != 0:
                    sd = shift_source(PE, tracked, [0] + [d] * 4 + [0] * 4, S)  # a+ window only
                    gd = (s_a(ad, enc, donor_tok, aplus, aminus, Dh, source=sd, tracked=tracked)
                          - s_a(ad, enc, donor_tok, aplus, aminus, Dh, source=sd, tracked=tracked,
                                ablate_head=ADDRESSER))
                    Gd[d] += gd.cpu().tolist()
                else:
                    Gd[d] += gc.cpu().tolist()
            if prob_inv is None:
                nat = s_a(ad, enc, donor_tok, aplus, aminus, Dh)
                s0 = shift_source(PE, tracked, [0] * 9, S)
                clamp0 = s_a(ad, enc, donor_tok, aplus, aminus, Dh, source=s0, tracked=tracked)
                shift0_err = float((clamp0 - nat).abs().max())
                o = ad.model(**enc, output_attentions=True)
                attn = o.attentions[L3][:, H3]
                proj = capture_proj(ad, L3, ["query", "key"], lambda: ad.model(**enc))
                q = proj["query"][:, :, H3 * Dh:(H3 + 1) * Dh]
                k = proj["key"][:, :, H3 * Dh:(H3 + 1) * Dh]
                add = (1.0 - enc["attention_mask"][:, None, :].float()) * -1e9
                recon = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / (Dh ** 0.5) + add, -1)
                qk_err = float((recon - attn).abs().max())
                rs = float((attn.sum(-1) - 1).abs().max())
                prob_inv = {"PASS": bool(rs < 1e-3 and float(attn.max()) > 0), "max_row_sum_err": rs}
            print(f"  {min(i + args.batch, len(pairs))}/{len(pairs)}", end="\r", flush=True)
    print()

    G0 = np.array(Gc[0]); G0_mean = float(np.mean(G0)); G0_ci = list(gboot(G0.tolist(), genes))
    checks = {
        "shift0_is_native": {"PASS": bool(shift0_err < 1e-4), "max_abs_err": shift0_err},
        "l2h15_contributes": {"PASS": bool(abs(G0_mean) > 0 and (G0_ci[0] > 0 or G0_ci[1] < 0)),
                              "G15_0_mean": G0_mean, "ci": G0_ci},
        "qk_reconstruction": {"PASS": bool(qk_err < 1e-3), "max_abs_err": qk_err},
        "probability_invariants": prob_inv,
        "grid_fits": {"PASS": bool(len(pairs) >= args.min_n or args.smoke), "n_kept": len(pairs)},
        "pe_table_size": {"PASS": True, "P": int(P)},
        "both_strands_present": {"PASS": len(strands) == 2, "strands": strands},
    }
    ckpt = Path(args.model); ckpt = ckpt if ckpt.is_absolute() else REPO / ckpt
    checks["checkpoint_recorded"] = {"PASS": True, "sha256": sha256_file(ckpt / "pytorch_model.bin")}
    pf = {"commit": git_commit(REPO), "n": len(pairs), "grid": list(GRID), "checks": checks,
          "PASS": bool(all(c["PASS"] for c in checks.values()))}
    pf_path.parent.mkdir(parents=True, exist_ok=True)
    pf_path.write_text(json.dumps(pf, indent=1))
    print(f"preflight PASS={pf['PASS']}  -> {pf_path}")
    for name, c in pf["checks"].items():
        print(f"    {'ok ' if c['PASS'] else 'XX '}{name}"
              + (f"  ({c.get('max_abs_err')})" if "max_abs_err" in c else ""))

    if args.smoke:
        print("SMOKE: preflight/memorials only; no E(Delta) curves reported (val sealed).")
        return
    if not pf["PASS"] and not args.force:
        print("  *** preflight FAILED -- not reporting curves (prereg §3).")
        return

    def curve(G):
        return {str(d): {"G15_mean": float(np.mean(G[d])),
                         "E_mean": float(np.mean(np.array(G[d]) - G0)),
                         "ci": list(gboot((np.array(G[d]) - G0).tolist(), genes))} for d in GRID}

    def slope(G):
        xs = np.repeat(np.array(GRID, float), len(G0))
        ys = np.concatenate([np.array(G[d]) for d in GRID])
        return float(np.polyfit(xs, ys, 1)[0])

    cc, cd = curve(Gc), curve(Gd)
    # DIAGNOSTIC: does the intact preference S_A survive the common shift, or collapse (artifact)?
    diag = {str(d): {"S_A_intact": float(np.mean(Sc_int[d])),
                     "S_A_ablated": float(np.mean(Sc_abl[d]))} for d in GRID}
    maxE_c = max(abs(cc[str(d)]["E_mean"]) for d in GRID)
    maxE_d = max(abs(cd[str(d)]["E_mean"]) for d in GRID)
    toe = toeplitz_fraction(PE, Wq, Wk, Dh)
    result = {
        "n": len(pairs), "n_genes": len(set(genes)), "strands": strands, "P": int(P),
        "grid": list(GRID), "G15_0": {"mean": G0_mean, "ci": G0_ci}, "E_common": cc, "E_diff": cd,
        "common_max_absE": maxE_c, "common_max_ratio": (maxE_c / abs(G0_mean)) if G0_mean else None,
        "common_slope_per_nt": slope(Gc), "diff_max_absE": maxE_d, "diff_slope_per_nt": slope(Gd),
        "diagnostic_SA_under_common": diag,
        "raw_pe_kernel_toeplitz_residual_fraction": toe, "preflight_PASS": pf["PASS"],
        "meets_power_target": bool(len(pairs) >= 200),
        "confirmatory": bool(pf["PASS"] and len(pairs) >= 200),
        "REFLECTION": {"status": "PENDING -- fill after inspection (prereg §5)"},
    }
    out.write_text(json.dumps(result, indent=1))
    status = "CONFIRMATORY" if result["confirmatory"] else "DIAGNOSTIC"
    man = ExperimentManifest.create(
        "splicebert_coupling:a4_address_geometry", sys.argv, status=status,
        config={"split": args.split, "n": len(pairs), "n_genes": result["n_genes"],
                "grid": list(GRID), "reader": list(READER), "addresser": list(ADDRESSER),
                "preflight_PASS": pf["PASS"], "meets_power_target": result["meets_power_target"]},
        seeds={"bootstrap": 0},
        notes=["A4: geometry of L2H15's address contribution G15 = S_A^intact - S_A^ablated (S_A = "
               "QK preference a+ vs decoy) under common vs differential PE translation. QK-level.",
               f"status={status} (preflight_PASS={pf['PASS']}, power={result['meets_power_target']})."])
    man.add_input(str(ckpt / "pytorch_model.bin"))
    for data in (args.gtf, args.fasta):
        if Path(data).exists():
            man.add_input(data)
    man.write(out.parent / (out.stem + ".manifest.json"))

    print("=" * 80)
    print(f"  G15(0) {G0_mean:+.3f} {G0_ci}")
    print("  COMMON (all offsets preserved):")
    for d in GRID:
        print(f"    D={d:+4d}  G15 {cc[str(d)]['G15_mean']:+.3f}  E {cc[str(d)]['E_mean']:+.3f} "
              f"[{cc[str(d)]['ci'][0]:+.3f},{cc[str(d)]['ci'][1]:+.3f}]")
    print(f"  common max|E| {maxE_c:.3f} ({result['common_max_ratio']:.2f} of |G15_0|)  "
          f"slope {result['common_slope_per_nt']:+.5f}/nt")
    print(f"  DIFFERENTIAL max|E| {maxE_d:.3f}  slope {result['diff_slope_per_nt']:+.5f}/nt   "
          f"raw-PE Toeplitz residual {toe:.3f}")
    print("  DIAGNOSTIC S_A under common (does the intact preference survive the shift?):")
    for d in GRID:
        print(f"    D={d:+4d}  S_A_intact {diag[str(d)]['S_A_intact']:+.3f}  "
              f"S_A_ablated {diag[str(d)]['S_A_ablated']:+.3f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
