#!/usr/bin/env python
"""Donor-graph experiment 4 -- A0: stage-wise provenance of L3H1's acceptor targeting.

Frozen protocol: studies/splicebert_partner_selection/PREREGISTRATION_A0_TARGETING_PROVENANCE.md.

At what depth in the residual supplied to L3H1 does the donor-query preference for the intron-mate
acceptor (over a phase/distance-matched real-acceptor decoy, primary; a generic AG decoy,
secondary) become SUFFICIENT, and is it carried by the KEY side, the QUERY side, or a Q x K
interaction?  Reconstruct L3's Q/K stage-by-stage from a hybrid residual whose stages after ``s``
are clamped to a mean-resample reference (post-LN => intervention, not linear attribution), read
three curves ``S_K/S_Q/S_J``, a reciprocal leave-one-stage-out necessity check, and an absolute-
attraction diagnostic.  ``S_A`` is a mechanistic intermediate: no behavioural partner-selection
claim (that task was withdrawn as degenerate).

The run writes a frozen preflight FIRST and stops if it fails.  ``--smoke`` runs ONLY the
arithmetic/preflight invariants and emits NO effect directions, keeping the val set sealed.
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
from gfmcd.rigor import empirical_upper_p_value  # noqa: E402
from gfmcd.splicing.address import build_pairs, match_audit, windows  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    ADDRESSER, DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, READER, CouplingTask, build_windows,
    capture_proj, gboot, gboot_ratio, load_adapter, load_examples, pick_device, ratio_of_means,
)
from gfmcd.splicing.provenance import (  # noqa: E402
    COMPONENTS, N_STAGES, RoleMean, attraction, capture_all, clamp_after, reconstruct_qk,
    scatter_reference, three_curves,
)

L3, H3 = READER
SPLIT_CHROMS = {"val": ("chr11", "chr16", "chr19"), "test": ("chr8", "chr9")}
STAGE_LABELS = ["embed", "attn0", "mlp0", "attn1", "mlp1", "attn2", "mlp2"]   # stages 1..7


def tracked_positions(ch, off, other_attr, device):
    """[B, 9] token indices per pair: donor, a+ window (4), a(other) window (4)."""
    donor = torch.tensor([[p.example.donor_pos + off] for p in ch], device=device)
    wplus = windows([p.a_plus for p in ch], off, device)
    wother = windows([getattr(p, other_attr) for p in ch], off, device)
    return torch.cat([donor, wplus, wother], dim=1), donor[:, 0], wplus, wother


# --------------------------------------------------------------------------------------
# Reference construction (pass 1): mean-resample role means over the whole arm.
# --------------------------------------------------------------------------------------
def build_reference(ad, task, pairs, off, other_attr, batch):
    acc = None
    with torch.no_grad():
        for i in range(0, len(pairs), batch):
            ch = pairs[i:i + batch]
            exs = [p.example for p in ch]
            enc = task.encode([x.sequence for x in exs], [x.donor_pos for x in exs])
            tp, *_ = tracked_positions(ch, off, other_attr, ad.device)
            captured = capture_all(ad, lambda: ad.model(**enc))
            if acc is None:
                acc = RoleMean(tp.shape[1], ad.hidden_size, ad.device)
            acc.update(captured, tp)
    return acc.means()


# --------------------------------------------------------------------------------------
# Reconstruction sweep (pass 2): three curves x 8 stages + LOO + attraction, per arm.
# --------------------------------------------------------------------------------------
def sweep_arm(ad, task, pairs, off, other_attr, means, batch):
    Dh = ad.hidden_size // ad.num_heads
    # per-example accumulators
    curves = {c: {s: [] for s in range(N_STAGES + 1)} for c in ("S_K", "S_Q", "S_J")}
    loo = {c: [] for c in COMPONENTS}                       # drop when component c -> reference
    # absolute-window attraction is reported PER CURVE (prereg §2): the query-config (q_s, k_F)
    # curve is what reveals a query effect that lifts the whole acceptor class vs background
    # while leaving a+ - a(other) flat -- a coupling contribution invisible to the difference.
    attr = {cfg: {s: {"a_plus_vs_bg": [], "a_other_vs_bg": []} for s in range(N_STAGES + 1)}
            for cfg in ("S_K", "S_Q", "S_J")}
    genes: list[str] = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch):
            ch = pairs[i:i + batch]
            exs = [p.example for p in ch]
            enc = task.encode([x.sequence for x in exs], [x.donor_pos for x in exs])
            S = enc["input_ids"].shape[1]
            tp, donor_tok, wplus, wother = tracked_positions(ch, off, other_attr, ad.device)
            wbg = build_windows(exs, off, ad.device).control
            ref = scatter_reference(means, tp, S, ad.device)
            genes += [p.gene_id for p in ch]

            q_F, k_F = reconstruct_qk(ad, enc, tp, (), ref)          # stage 7 == clean
            for s in range(N_STAGES + 1):                            # 0..7
                q_s, k_s = (q_F, k_F) if s == N_STAGES else reconstruct_qk(
                    ad, enc, tp, clamp_after(s), ref)
                cur = three_curves(q_s, k_s, q_F, k_F, donor_tok, wplus, wother, Dh)
                for c in curves:
                    curves[c][s] += cur[c].tolist()
                for cfg, (qq, kk) in (("S_K", (q_F, k_s)), ("S_Q", (q_s, k_F)),
                                      ("S_J", (q_s, k_s))):
                    a = attraction(qq, kk, donor_tok, wplus, wother, wbg, Dh)
                    attr[cfg][s]["a_plus_vs_bg"] += a["a_plus_vs_bg"].tolist()
                    attr[cfg][s]["a_other_vs_bg"] += a["a_other_vs_bg"].tolist()
            # reciprocal leave-one-stage-out: full clean minus one component -> reference
            full = np.array(three_curves(q_F, k_F, q_F, k_F, donor_tok, wplus, wother, Dh)["S_J"])
            for comp in COMPONENTS:
                q_g, k_g = reconstruct_qk(ad, enc, tp, (comp,), ref)
                sj = three_curves(q_g, k_g, q_F, k_F, donor_tok, wplus, wother, Dh)["S_J"]
                loo[comp] += (full - sj).tolist()
            print(f"  {min(i + batch, len(pairs))}/{len(pairs)}", end="\r", flush=True)
    print()
    return curves, loo, attr, genes


def smoke_selfcheck(ad, task, pairs, off, means, batch):
    """Model-level memorial that leaks NO effect direction: run the full sweep on a handful of
    pairs and assert the three-curve endpoint identity ``S_K(7)=S_Q(7)=S_J(7)`` and finiteness."""
    sub = pairs[: min(6, len(pairs))]
    cur, loo, _attr, _g = sweep_arm(ad, task, sub, off, "a_minus", means, batch)
    j7 = np.array(cur["S_J"][N_STAGES])
    endpoint_err = max(float(np.abs(np.array(cur["S_K"][N_STAGES]) - j7).max()),
                       float(np.abs(np.array(cur["S_Q"][N_STAGES]) - j7).max()))
    finite = (all(np.isfinite(cur[c][s]).all() for c in cur for s in cur[c])
              and all(np.isfinite(loo[k]).all() for k in loo))
    return endpoint_err, bool(finite)


def summarise(curves, loo, attr, genes, min_denom):
    def ci(v):
        return list(gboot(v, genes))
    dnm = curves["S_J"][N_STAGES]                            # S_J(7) == S_A per example
    denom_mean = float(np.mean(dnm))
    out = {"S_A_full": {"mean": denom_mean, "ci": ci(dnm)}, "curves": {}, "normalized": {},
           "leave_one_out": {}, "attraction": {}, "min_denom": min_denom,
           "normalized_reported": bool(abs(denom_mean) > min_denom)}
    for c in ("S_K", "S_Q", "S_J"):
        out["curves"][c] = [{"stage": s, "label": (["all-ref"] + STAGE_LABELS)[s],
                             "mean": float(np.mean(curves[c][s])), "ci": ci(curves[c][s])}
                            for s in range(N_STAGES + 1)]
        if out["normalized_reported"]:
            out["normalized"][c] = [
                {"stage": s, "ratio": ratio_of_means(curves[c][s], dnm),
                 "ci": list(gboot_ratio(curves[c][s], dnm, genes))}
                for s in range(N_STAGES + 1)]
    for comp, lbl in zip(COMPONENTS, ["embed"] + STAGE_LABELS[1:]):
        v = loo[comp]
        key = "embed" if comp == "embed" else f"{comp[0]}:{comp[1]}"
        out["leave_one_out"][key] = {"drop_mean": float(np.mean(v)), "ci": ci(v)}
    for cfg in ("S_K", "S_Q", "S_J"):
        out["attraction"][cfg] = [
            {"stage": s, "label": (["all-ref"] + STAGE_LABELS)[s],
             "a_plus_vs_bg": float(np.mean(attr[cfg][s]["a_plus_vs_bg"])),
             "a_other_vs_bg": float(np.mean(attr[cfg][s]["a_other_vs_bg"]))}
            for s in range(N_STAGES + 1)]
    return out


# --------------------------------------------------------------------------------------
# Preflight: the arithmetic memorials that lock the reconstruction (prereg §5).
# --------------------------------------------------------------------------------------
def preflight(ad, task, pairs, off, args, means_secondary, n_real: int) -> dict:
    frozen = {"attn_row_sum_tol": 1e-3, "qk_recon_tol": 1e-3,
              "positionwise_tol": 1e-4, "stage7_tol": 1e-5, "min_real": args.min_real}
    Dh = ad.hidden_size // ad.num_heads
    checks: dict = {}
    sub = pairs[: min(len(pairs), 32)]
    exs = [p.example for p in sub]
    dn = [x.donor_pos for x in exs]
    with torch.no_grad():
        enc = task.encode([x.sequence for x in exs], dn)
        S = enc["input_ids"].shape[1]
        o = ad.model(**enc, output_attentions=True)
        attn = o.attentions[L3][:, H3]
        row_sums = attn.sum(-1)
        checks["probability_invariants"] = {
            "PASS": bool(float((row_sums - 1).abs().max()) < frozen["attn_row_sum_tol"]
                         and float(attn.abs().max()) > 0),
            "max_row_sum_err": float((row_sums - 1).abs().max())}
        # G-QK: analytic softmax(q.k/sqrt d) reproduces the model's own L3H1 attention
        proj = capture_proj(ad, L3, ["query", "key"], lambda: ad.model(**enc))
        q = proj["query"][:, :, H3 * Dh:(H3 + 1) * Dh]
        k = proj["key"][:, :, H3 * Dh:(H3 + 1) * Dh]
        add = (1.0 - enc["attention_mask"][:, None, :].float()) * -1e9
        recon = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / (Dh ** 0.5) + add, dim=-1)
        qk_err = float((recon - attn).abs().max())
        checks["qk_reconstruction"] = {"PASS": bool(qk_err < frozen["qk_recon_tol"]),
                                       "max_abs_err": qk_err}

        tp, *_ = tracked_positions(sub, off, "a_minus", ad.device)
        ref = scatter_reference(means_secondary, tp, S, ad.device)
        # stage7 == clean: reconstruct with no clamp equals the raw clean projections
        q7, k7 = reconstruct_qk(ad, enc, tp, (), ref)
        s7 = max(float((q7 - proj["query"]).abs().max()), float((k7 - proj["key"]).abs().max()))
        checks["stage7_equals_clean"] = {"PASS": bool(s7 < frozen["stage7_tol"]),
                                         "max_abs_err": s7}
        # positionwise: clamping ONLY the tracked positions gives the same L3 Q/K at those
        # positions as clamping ALL positions (validates the positionwise-LN restriction; the
        # untracked clamp values are provably irrelevant at tracked positions if it holds).
        mid = clamp_after(3)                                  # a stage with real clamping
        q_a, k_a = reconstruct_qk(ad, enc, tp, mid, ref)      # tracked-only clamp
        allpos = torch.arange(S, device=ad.device)[None].expand(len(sub), S)
        q_b, k_b = reconstruct_qk(ad, enc, allpos, mid, ref)  # all-positions clamp (same ref)
        b = torch.arange(len(sub))[:, None].expand_as(tp)
        pw = max(float((q_a[b, tp] - q_b[b, tp]).abs().max()),
                 float((k_a[b, tp] - k_b[b, tp]).abs().max()))
        checks["positionwise_reconstruction"] = {"PASS": bool(pw < frozen["positionwise_tol"]),
                                                 "max_abs_err": pw}
    # structural tripwire (falsifiable): the partner AG must be INTACT at every a+ in the set the
    # reference is built from.  A corrupt run (AG->CC at a+-2) would read "CC" here; this fails
    # hard if the rejected corrupt-run reference ever leaked in (prereg §3, D-A0-1).
    ag_intact = [p.example.sequence[p.a_plus - 2:p.a_plus] == "AG" for p in pairs]
    frac_ag = float(np.mean(ag_intact))
    checks["reference_is_clean_sequence"] = {
        "PASS": bool(frac_ag >= 0.95), "frac_partner_AG_intact": frac_ag,
        "note": "a+ AG intact in the reference set (would be CC under the rejected corrupt reference)"}
    checks["resolution_pair_match"] = {"PASS": bool(match_audit(pairs)["PASS"]), **match_audit(pairs)}
    checks["real_acceptor_arm_available"] = {"PASS": bool(n_real >= frozen["min_real"]),
                                             "n_real": n_real, "min_real": frozen["min_real"]}
    checks["both_strands_present"] = {"PASS": len({p.strand for p in pairs}) == 2,
                                      "strands": sorted({p.strand for p in pairs})}
    ckpt = Path(args.model); ckpt = ckpt if ckpt.is_absolute() else REPO / ckpt
    checks["checkpoint_recorded"] = {"PASS": True, "sha256": sha256_file(ckpt / "pytorch_model.bin")}
    PASS = all(c["PASS"] for c in checks.values())
    return {"frozen": frozen, "commit": git_commit(REPO), "n_pairs": len(pairs), "n_real": n_real,
            "checks": checks, "PASS": bool(PASS)}


# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=260)
    ap.add_argument("--pool", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tol", type=int, default=40)
    ap.add_argument("--real-tol", type=int, default=60)
    ap.add_argument("--min-real", type=int, default=30, help="min real-acceptor pairs (primary arm)")
    ap.add_argument("--real-cap", type=int, default=120,
                    help="max real-acceptor pairs (primary arm), drawn from the FULL screened pool")
    ap.add_argument("--min-denom", type=float, default=0.15, help="floor on |S_A| for normalized")
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--skip-coding", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="arithmetic/preflight invariants ONLY; emit NO effect directions (seals val)")
    ap.add_argument("--force", action="store_true",
                    help="compute curves even if preflight fails (DIAGNOSTIC only)")
    ap.add_argument("--out", default="results/splicebert_coupling/a0_targeting_provenance.json")
    args = ap.parse_args()

    out = REPO / args.out
    pf_path = out.parent / "preflight_a0.json"
    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=True)
    task = CouplingTask(ad, batch=args.batch)
    off = task.off

    Xpool, _ = load_examples(args.gtf, args.fasta, args.split, args.pool, task=task,
                             tag=f"a0-{args.split}", min_effect=None)
    cds = None
    if not args.skip_coding:
        try:
            from gfmcd.splicing.phase import load_cds_frames
            cds = load_cds_frames(args.gtf, allowed_chromosomes=SPLIT_CHROMS.get(args.split))
        except Exception as e:
            print(f"  (coding covariate unavailable: {e})")
    avail = build_pairs(Xpool, tol=args.tol, real_tol=args.real_tol, cds_lookup=cds)
    n = min(args.n, 24) if args.smoke else args.n
    pairs = avail[:n]                                        # secondary / generic-AG arm
    if len(pairs) < 8:
        raise RuntimeError(f"only {len(avail)} phase-matched pairs from {len(Xpool)} eligible; "
                           "raise --pool or --tol.")
    # primary arm draws real-acceptor pairs from the FULL screened pool (prereg §6 amendment),
    # capped, so it is as powered as the sparse real-acceptor yield allows.
    real_cap = min(args.real_cap, 12) if args.smoke else args.real_cap
    real_pairs = [p for p in avail if p.a_real is not None][:real_cap]
    print(f"{len(pairs)} generic pairs, {len(real_pairs)} real-acceptor pairs (from {len(avail)} "
          f"available); strands={sorted({p.strand for p in pairs})}; device={dev}"
          f"{'  [SMOKE]' if args.smoke else ''}")

    # reference for the secondary (a-) arm is needed by the preflight memorials
    means_secondary = build_reference(ad, task, pairs, off, "a_minus", args.batch)
    pf = preflight(ad, task, pairs, off, args, means_secondary, n_real=len(real_pairs))
    pf_path.parent.mkdir(parents=True, exist_ok=True)
    pf_path.write_text(json.dumps(pf, indent=1))
    print(f"preflight PASS={pf['PASS']}  -> {pf_path}")
    for name, c in pf["checks"].items():
        print(f"    {'ok ' if c['PASS'] else 'XX '}{name}"
              + (f"  ({c.get('max_abs_err')})" if "max_abs_err" in c else ""))

    if args.smoke:
        err, finite = smoke_selfcheck(ad, task, pairs, off, means_secondary, args.batch)
        ok = bool(err < 1e-4 and finite)
        print(f"    {'ok ' if ok else 'XX '}selfcheck  endpoint_identity_err={err:.2e}  "
              f"finite={finite}")
        print("SMOKE: preflight/arithmetic + endpoint-identity only; no curves reported "
              "(val stays sealed).")
        return
    if not pf["PASS"] and not args.force:
        print("  *** preflight FAILED -- not computing curves (prereg §5).")
        return
    if not pf["PASS"]:
        print("  *** preflight FAILED but --force: curves below are DIAGNOSTIC ONLY.")

    # ----- primary arm (real-acceptor contrast) + secondary arm (generic-AG) ---------------
    result: dict = {"n_pairs": len(pairs), "n_real": len(real_pairs),
                    "strands": sorted({p.strand for p in pairs}),
                    "match_audit": match_audit(pairs), "preflight_PASS": pf["PASS"]}
    if len(real_pairs) >= 8:
        print(f"primary arm (a+ vs real acceptor), n={len(real_pairs)}")
        means_p = build_reference(ad, task, real_pairs, off, "a_real", args.batch)
        cur, loo, attrp, gp = sweep_arm(ad, task, real_pairs, off, "a_real", means_p, args.batch)
        result["primary_real_acceptor_contrast"] = summarise(cur, loo, attrp, gp, args.min_denom)
        # the primary arm's own distance confound (a_real sits farther from the donor on average);
        # reported on THIS set, not the secondary pairs (prereg §1/§4 -- signed gap adjudicates
        # identity vs distance-decay). n_genes for the primary arm reported alongside.
        result["primary_real_acceptor_contrast"]["real_distance_confound"] = \
            match_audit(real_pairs)["real_acceptor_distance"]
        result["primary_real_acceptor_contrast"]["n_genes"] = len(set(p.gene_id for p in real_pairs))
    else:
        result["primary_real_acceptor_contrast"] = {"n": len(real_pairs), "note": "< 8, skipped"}
    print(f"secondary arm (a+ vs generic AG), n={len(pairs)}")
    cur2, loo2, attr2, g2 = sweep_arm(ad, task, pairs, off, "a_minus", means_secondary, args.batch)
    result["secondary_generic_contrast"] = summarise(cur2, loo2, attr2, g2, args.min_denom)

    result["n_genes"] = len(set(p.gene_id for p in pairs))
    result["meets_power_target"] = bool(len(pairs) >= 250 and len(real_pairs) >= args.min_real)
    result["confirmatory"] = bool(pf["PASS"] and result["meets_power_target"])
    result["REFLECTION"] = {"status": "PENDING -- fill after inspection (prereg §7)"}

    out.write_text(json.dumps(result, indent=1))
    status = "CONFIRMATORY" if result["confirmatory"] else "DIAGNOSTIC"
    man = ExperimentManifest.create(
        "splicebert_coupling:a0_targeting_provenance", sys.argv, status=status,
        config={"split": args.split, "n_pairs": len(pairs), "n_real": len(real_pairs),
                "n_genes": result["n_genes"], "reader": list(READER), "addresser": list(ADDRESSER),
                "preflight_PASS": pf["PASS"], "meets_power_target": result["meets_power_target"],
                "min_denom": args.min_denom},
        seeds={"bootstrap": 0},
        notes=["Stage-wise SUFFICIENCY of L3H1's donor-query acceptor targeting (S_K/S_Q/S_J) via "
               "mean-resample reconstruction; reciprocal leave-one-stage-out for necessity.",
               "S_A is a mechanistic intermediate: NO behavioural partner-selection claim.",
               "Real-acceptor contrast is primary (strength-controlled, not identity-clean); "
               "generic-AG secondary.",
               f"status={status} (preflight_PASS={pf['PASS']}, power={result['meets_power_target']})."])
    ckpt = Path(args.model); ckpt = ckpt if ckpt.is_absolute() else REPO / ckpt
    man.add_input(str(ckpt / "pytorch_model.bin"))
    for data in (args.gtf, args.fasta):
        if Path(data).exists():
            man.add_input(data)
    man.write(out.parent / (out.stem + ".manifest.json"))

    print("=" * 80)
    for arm in ("primary_real_acceptor_contrast", "secondary_generic_contrast"):
        r = result[arm]
        if "curves" not in r:
            print(f"{arm}: {r.get('note')}")
            continue
        sj = r["curves"]["S_J"]
        print(f"{arm}: S_A(full)={r['S_A_full']['mean']:+.3f}  S_J by stage: "
              + " ".join(f"{d['label']}={d['mean']:+.2f}" for d in sj))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
