#!/usr/bin/env python
"""Donor-graph experiment 15 -- A11: is L3H1's address ABSOLUTE or RELATIVE? (A4 redesign).

Frozen protocol: studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_A11_ADDRESS_GEOMETRY.md.

A4 clamped ISOLATED-token position embeddings -> inconsistent field -> S_A collapsed (confound).  A11
uses a CONSISTENT GLOBAL position_ids shift on a TRUNCATED window (W=640, PE-table headroom), with a
MANDATORY intact-S_A-survival memorial.  E_common(D) = S_A(D) - S_A(0); near-invariance => RELATIVE,
systematic => ABSOLUTE.  Reuses the A1a/A4 S_A instrument.  --smoke = memorials only (val sealed).
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

from gfmcd.provenance import ExperimentManifest, git_commit, sha256_file       # noqa: E402
from gfmcd.splicing.address import phase_matched_decoy, qk_logit_to_window, windows  # noqa: E402
from gfmcd.splicing.circuit import (                                          # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, READER, CouplingTask, capture_proj, gboot,
    load_adapter, load_examples, pick_device,
)
from gfmcd.splicing.provenance import embed_component                          # noqa: E402

L3, H3 = READER
OFFS = (-3, -2, -1, 0)
GRID = (-192, -128, -64, 0, 64, 128, 192)
W = 640
BASE = 192


def tracked(pairs, off, device):
    donor = torch.tensor([[x.donor_pos + off] for x, _ in pairs], device=device)[:, 0]
    aplus = windows([x.next_acceptor_pos for x, _ in pairs], off, device, offsets=OFFS)
    aminus = windows([am for _, am in pairs], off, device, offsets=OFFS)
    return donor, aplus, aminus


def s_a(ad, enc, donor_tok, aplus, aminus, Dh, pid=None):
    def run():
        return ad.model(**enc, position_ids=pid) if pid is not None else ad.model(**enc)
    proj = capture_proj(ad, L3, ["query", "key"], run)
    q, k = proj["query"], proj["key"]
    return (qk_logit_to_window(q, k, H3, Dh, donor_tok, aplus)
            - qk_logit_to_window(q, k, H3, Dh, donor_tok, aminus))


def toeplitz_fraction(PE, Wq, Wk, Dh, lo=100, hi=760, step=8):
    idx = torch.arange(lo, hi, step, device=PE.device)
    K = ((PE[idx] @ Wq.T) @ (PE[idx] @ Wk.T).T) / (Dh ** 0.5)
    G = K.shape[0]
    resid = [torch.diagonal(K, offset=d) - torch.diagonal(K, offset=d).mean()
             for d in range(-(G - 1), G) if torch.diagonal(K, offset=d).numel() >= 3]
    within = float((torch.cat(resid) ** 2).mean()); total = float(((K - K.mean()) ** 2).mean())
    return within / total if total else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val"); ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--min-n", type=int, default=150); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tol", type=int, default=40); ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF); ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="results/splicebert_coupling/a11_address_geometry.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    out = REPO / args.out; pf_path = out.parent / "preflight_a11_address_geometry.json"
    dev = pick_device(args.device); ad = load_adapter(args.model, dev, eager=True)
    task = CouplingTask(ad, batch=args.batch); off = task.off
    Dh = ad.hidden_size // ad.num_heads
    PE = embed_component(ad, "position").weight.detach(); P = PE.shape[0]
    Wq = ad.attention_self_modules[L3].query.weight[H3 * Dh:(H3 + 1) * Dh, :].detach()
    Wk = ad.attention_self_modules[L3].key.weight[H3 * Dh:(H3 + 1) * Dh, :].detach()

    Xpool, _ = load_examples(args.gtf, args.fasta, args.split, args.n, task=task, tag=f"a11-{args.split}",
                             min_effect=None)

    def fits(x, am):        # all tracked tokens fit in the W-window with margin
        toks = [x.donor_pos + off] + [x.next_acceptor_pos + off + k for k in OFFS] + [am + off + k for k in OFFS]
        return max(toks) < W - 8 and min(toks) >= 0

    pairs = []
    for x in Xpool:
        am = phase_matched_decoy(x, tol=args.tol)
        if am is not None and fits(x, am):
            pairs.append((x, am))
    if args.smoke:
        pairs = pairs[:24]
    if len(pairs) < 8:
        raise RuntimeError(f"only {len(pairs)} decoy+W-fitting pairs of {len(Xpool)}.")
    strands = sorted({x.strand for x, _ in pairs}); genes = [x.gene_id for x, _ in pairs]
    print(f"{len(pairs)} decoy+W-fitting pairs (of {len(Xpool)}); P={P}; W={W}; BASE={BASE}")

    SA = {d: [] for d in GRID}
    noop_err = 0.0; sa_full, sa_trunc = [], []
    for i in range(0, len(pairs), args.batch):
        ch = pairs[i:i + args.batch]
        exs = [x for x, _ in ch]; dn = [x.donor_pos for x in exs]
        enc_full = task.encode([x.sequence for x in exs], dn)
        B = enc_full["input_ids"].shape[0]
        enc = {"input_ids": enc_full["input_ids"][:, :W].contiguous(),
               "attention_mask": enc_full["attention_mask"][:, :W].contiguous()}
        donor_tok, aplus, aminus = tracked(ch, off, dev)
        for d in GRID:
            pid = (torch.arange(W, device=dev) + BASE + d)[None].expand(B, W)
            SA[d] += s_a(ad, enc, donor_tok, aplus, aminus, Dh, pid=pid).cpu().tolist()
        # memorials (accumulate on all batches)
        nat = s_a(ad, enc, donor_tok, aplus, aminus, Dh, pid=None)                # truncated native
        pid_nat = torch.arange(W, device=dev)[None].expand(B, W)
        via_pid = s_a(ad, enc, donor_tok, aplus, aminus, Dh, pid=pid_nat)         # position_ids no-op
        noop_err = max(noop_err, float((via_pid - nat).abs().max()))
        full = s_a(ad, enc_full, donor_tok, aplus, aminus, Dh, pid=None)          # full-length
        sa_full += full.cpu().tolist(); sa_trunc += nat.cpu().tolist()
        print(f"  {min(i + args.batch, len(pairs))}/{len(pairs)}", end="\r", flush=True)
    print()

    SA0 = np.array(SA[0]); SA0_mean = float(np.mean(SA0))
    surv = {str(d): float(np.mean(SA[d])) for d in GRID}
    survive_ok = bool(SA0_mean >= 0.5 and all(np.mean(SA[d]) >= 0.5 * SA0_mean for d in GRID))
    trunc_corr = float(np.corrcoef(sa_full, sa_trunc)[0, 1]) if len(sa_full) > 2 else float("nan")

    checks = {
        "shift_is_noop_at_native": {"PASS": bool(noop_err < 1e-4), "max_abs_err": noop_err},
        "intact_S_A_survives": {"PASS": survive_ok, "S_A_0": SA0_mean, "S_A_by_delta": surv},
        "truncation_preserves_targeting": {"PASS": bool(np.mean(sa_trunc) > 0 and trunc_corr > 0.5),
                                           "S_A_trunc_mean": float(np.mean(sa_trunc)),
                                           "full_vs_trunc_corr": trunc_corr},
        "grid_fits": {"PASS": bool(len(pairs) >= args.min_n or args.smoke), "n_kept": len(pairs)},
        "both_strands_present": {"PASS": len(strands) == 2, "strands": strands},
    }
    ckpt = Path(args.model); ckpt = ckpt if ckpt.is_absolute() else REPO / ckpt
    checks["checkpoint_recorded"] = {"PASS": True, "sha256": sha256_file(ckpt / "pytorch_model.bin")}
    pf = {"commit": git_commit(REPO), "n": len(pairs), "grid": list(GRID), "checks": checks}
    pf["PASS"] = bool(all(c["PASS"] for c in checks.values()))
    pf_path.parent.mkdir(parents=True, exist_ok=True); pf_path.write_text(json.dumps(pf, indent=1))
    print(f"preflight PASS={pf['PASS']}  -> {pf_path}")
    for name, c in pf["checks"].items():
        print(f"    {'ok ' if c['PASS'] else 'XX '}{name}" + (f"  ({c.get('max_abs_err')})" if "max_abs_err" in c else ""))

    if args.smoke:
        print("SMOKE: memorials only (val sealed)."); return
    if not pf["PASS"] and not args.force:
        print("  *** preflight FAILED (survival memorial?) -- not reporting E(Delta) (prereg s3)."); return

    E = {str(d): {"S_A_mean": float(np.mean(SA[d])),
                  "E_mean": float(np.mean(np.array(SA[d]) - SA0)),
                  "ci": list(gboot((np.array(SA[d]) - SA0).tolist(), genes))} for d in GRID}
    ds = np.array(GRID, float); ys = np.array([np.mean(SA[d]) for d in GRID])
    slope = float(np.polyfit(ds, ys, 1)[0])
    # slope CI via gene-bootstrap of per-example S_A vs delta
    rng = np.random.default_rng(0); ug = sorted(set(genes)); by = {g: [k for k, gg in enumerate(genes) if gg == g] for g in ug}
    sl = []
    SAarr = {d: np.array(SA[d]) for d in GRID}
    for _ in range(400):
        idx = np.concatenate([by[g] for g in rng.choice(ug, len(ug))])
        yy = np.array([SAarr[d][idx].mean() for d in GRID])
        sl.append(float(np.polyfit(ds, yy, 1)[0]))
    slope_ci = [float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5))]
    maxE = max(abs(E[str(d)]["E_mean"]) for d in GRID); rel = maxE / abs(SA0_mean)
    slope_excludes_0 = bool(slope_ci[0] > 0 or slope_ci[1] < 0)
    if rel < 0.20 and not slope_excludes_0:
        verdict = "RELATIVE"
    elif rel > 0.20 and slope_excludes_0:
        verdict = "ABSOLUTE"
    else:
        verdict = "MIXED_or_inconclusive"
    toe = toeplitz_fraction(PE, Wq, Wk, Dh)

    result = {"n": len(pairs), "n_genes": len(ug), "strands": strands, "P": int(P), "W": W, "BASE": BASE,
              "grid": list(GRID), "S_A_0": SA0_mean, "E_common": E,
              "max_abs_E_over_S_A0": rel, "slope": slope, "slope_ci": slope_ci,
              "slope_excludes_0": slope_excludes_0, "verdict": verdict,
              "truncation_full_vs_trunc_corr": trunc_corr, "toeplitz_residual_fraction": toe,
              "preflight_PASS": pf["PASS"], "meets_power_target": bool(len(pairs) >= args.min_n),
              "confirmatory": bool(pf["PASS"] and len(pairs) >= args.min_n),
              "REFLECTION": {"status": "PENDING -- fill after inspection (prereg s4)"}}
    out.write_text(json.dumps(result, indent=1))
    status = "CONFIRMATORY" if result["confirmatory"] else "DIAGNOSTIC"
    man = ExperimentManifest.create(
        "splicebert_coupling:a11_address_geometry", sys.argv, status=status,
        config={"split": args.split, "n": len(pairs), "n_genes": len(ug), "W": W, "BASE": BASE,
                "grid": list(GRID), "preflight_PASS": pf["PASS"]},
        seeds={"gene_bootstrap": 0},
        notes=["A11 (A4 redesign): absolute-vs-relative address via a consistent GLOBAL position_ids "
               "shift on a truncated window; intact-S_A-survival memorial. No new kernel.",
               f"status={status}, verdict={verdict}."])
    man.add_input(str(ckpt / "pytorch_model.bin")); man.write(out.parent / (out.stem + ".manifest.json"))

    print("=" * 84)
    print(f"  S_A(0)={SA0_mean:+.3f}  max|E|/|S_A0|={rel:.3f}  slope={slope:+.4f}/nt {slope_ci}  toeplitz={toe:.3f}")
    print("  E_common(D): " + "  ".join(f"{d:+d}:{E[str(d)]['E_mean']:+.3f}" for d in GRID))
    print(f"  VERDICT: {verdict}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
