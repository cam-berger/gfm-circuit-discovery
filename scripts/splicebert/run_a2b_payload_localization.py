#!/usr/bin/env python
"""Donor-graph experiment 6 -- A2b-(1): localize L3H1's value payload to U_info.

Frozen protocol: studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_A2b_PAYLOAD_LOCALIZATION.md.

With the clean attention row A+ held fixed (post-LN linearity: V = W_V r), reconstruct L3H1's donor
head output analytically for each residual-window payload and inject it (donor only), reading the
native margin recovery.  Tests whether the partner payload is the C16/C17 U_info subspace vs an
energy-matched random rank-8 distribution, plus acceptor/example specificity and necessity.

Preflight/memorials FIRST (the value_linearity memorial licenses the whole analytic approach); the
run stops on failure.  --smoke = memorials only (val sealed).
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
from gfmcd.circuits.hf_bert import capture_head_outputs  # noqa: E402
from gfmcd.rigor import empirical_upper_p_value  # noqa: E402
from gfmcd.splicing.address import windows  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    CORRUPT_ALT, DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, READER, CouplingTask, build_windows,
    capture_proj, gboot_ratio, load_adapter, load_examples, pick_device, ratio_of_means,
)
from gfmcd.splicing.coupling import substitute  # noqa: E402
from gfmcd.splicing.payload import (  # noqa: E402
    deliver, delivered_energy, head_output, inject_margin_donor, projector, value_weight_head,
)
from gfmcd.splicing.subspace import random_subspace, top_k_subspace  # noqa: E402

L3, H3 = READER
K_UINFO = 8
SEED = 0
OFFS = (-3, -2, -1, 0)
SPLIT_CHROMS = {"val": ("chr11", "chr16", "chr19"), "test": ("chr8", "chr9")}


def corrupt_at(task, exs):
    seqs = [substitute(x.sequence, x.next_acceptor_pos - 2, CORRUPT_ALT) for x in exs]
    return task.encode(seqs, [x.donor_pos for x in exs])


def gather_win(t, win):
    """t [B,S,D], win [B,K] -> [B,K,D]."""
    b = torch.arange(t.shape[0], device=t.device)[:, None].expand_as(win)
    return t[b, win]


def gather_row(row, win):
    """row [B,S], win [B,K] -> [B,K]."""
    b = torch.arange(row.shape[0], device=row.device)[:, None].expand_as(win)
    return row[b, win]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-rand", type=int, default=40, help="energy-matched random rank-8 subspaces")
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="results/splicebert_coupling/a2b_payload_localization.json")
    args = ap.parse_args()

    out = REPO / args.out
    pf_path = out.parent / "preflight_a2b_payload_localization.json"
    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=True)
    task = CouplingTask(ad, batch=args.batch)
    off = task.off
    Dh = ad.hidden_size // ad.num_heads
    Wv = value_weight_head(ad).to(ad.device)                 # [Dh, hidden]

    n = min(args.n, 24) if args.smoke else args.n
    n_rand = min(args.n_rand, 4) if args.smoke else args.n_rand
    X, _ = load_examples(args.gtf, args.fasta, args.split, n, task=task, tag=f"a2b-{args.split}",
                         min_effect=None)
    if len(X) < 8:
        raise RuntimeError(f"only {len(X)} eligible examples.")
    strands = sorted({x.strand for x in X})
    genes = [x.gene_id for x in X]
    print(f"{len(X)} eligible examples; strands={strands}; device={dev}"
          f"{'  [SMOKE]' if args.smoke else ''}")

    # ---- capture pass: small analytic ingredients per example --------------------------------
    base_L, Aacc_L, Actl_L, dracc_L, drctl_L = [], [], [], [], []
    Hcl_L, Hco_L, mcl_L, mco_L, dtok_L = [], [], [], [], []
    lin_err = 0.0                                            # value_linearity memorial (first batch)
    inject_id_err = 0.0                                     # inject A-.V- reproduces m_co
    prob_inv = None
    with torch.no_grad():
        for i in range(0, len(X), args.batch):
            exs = X[i:i + args.batch]
            dn = [x.donor_pos for x in exs]
            dtok = torch.tensor([d + off for d in dn], device=ad.device)
            b = torch.arange(len(exs), device=ad.device)
            clean = task.encode([x.sequence for x in exs], dn)
            corr = corrupt_at(task, exs)
            acc = windows([x.next_acceptor_pos for x in exs], off, ad.device, offsets=OFFS)
            ctl = build_windows(exs, off, ad.device).control

            o_cl = ad.model(**clean, output_hidden_states=True, output_attentions=True)
            A_row = o_cl.attentions[L3][:, H3][b, dtok]       # [B,S] donor attention row
            r_cl = o_cl.hidden_states[3]                      # L3-input residual [B,S,hidden]
            Hcl = o_cl.hidden_states[-1][b, dtok]
            m_cl = task.margin(o_cl.logits.float(), dn).cpu().numpy()
            o_co = ad.model(**corr, output_hidden_states=True)
            r_co = o_co.hidden_states[3]
            Hco = o_co.hidden_states[-1][b, dtok]
            m_co = task.margin(o_co.logits.float(), dn).cpu().numpy()
            V_co = capture_proj(ad, L3, ["value"], lambda: ad.model(**corr))["value"]
            V_co_h = V_co[:, :, H3 * Dh:(H3 + 1) * Dh]        # [B,S,Dh]

            base = torch.einsum("bs,bsd->bd", A_row, V_co_h)  # A+_d . V-   [B,Dh]
            dr_acc = gather_win(r_cl - r_co, acc)             # Δr at acceptor [B,K,hidden]
            dr_ctl = gather_win(r_cl - r_co, ctl)
            base_L.append(base); Aacc_L.append(gather_row(A_row, acc)); Actl_L.append(gather_row(A_row, ctl))
            dracc_L.append(dr_acc); drctl_L.append(dr_ctl)
            Hcl_L.append(Hcl.float().cpu().numpy()); Hco_L.append(Hco.float().cpu().numpy())
            mcl_L += m_cl.tolist(); mco_L += m_co.tolist(); dtok_L.append(dtok)

            if prob_inv is None:
                full_row = o_cl.attentions[L3][:, H3][b, dtok]
                rserr = float((full_row.sum(-1) - 1).abs().max())
                prob_inv = {"PASS": bool(rserr < 1e-3 and float(full_row.max()) > 0),
                            "max_row_sum_err": rserr}
                # value_linearity memorial: V+_acc - V-_acc == W_V (r+ - r-)_acc
                V_cl = capture_proj(ad, L3, ["value"], lambda: ad.model(**clean))["value"]
                V_cl_h = V_cl[:, :, H3 * Dh:(H3 + 1) * Dh]
                dV_true = gather_win(V_cl_h - V_co_h, acc)    # [B,K,Dh]
                dV_lin = deliver(Wv, dr_acc)
                lin_err = float((dV_true - dV_lin).abs().max())
                # inject_identity: injecting A-.V- (the model's own corrupt head output) reproduces
                # m_co -- the donor-only head patch is faithful.  (NOTE: o_d(full) restores the
                # clean values only at the ACCEPTOR WINDOW, corrupt elsewhere, so it deliberately
                # does NOT equal A+.V+; the corruption perturbs non-acceptor values L3H1 also reads.
                # value_linearity + inject_identity are the correct validators of the analytic form.)
                head_co = capture_head_outputs(ad, lambda: ad.model(**corr))[L3][b, H3, dtok]
                m_id = inject_margin_donor(ad, task, corr, dn, dtok, head_co)
                inject_id_err = float(np.abs(m_id - m_co).max())
            print(f"  capture {min(i + args.batch, len(X))}/{len(X)}", end="\r", flush=True)
    print()

    base = torch.cat(base_L); Aacc = torch.cat(Aacc_L); Actl = torch.cat(Actl_L)
    dracc = torch.cat(dracc_L); drctl = torch.cat(drctl_L); dtok_all = torch.cat(dtok_L)
    Hcl = np.concatenate(Hcl_L); Hco = np.concatenate(Hco_L)
    mcl = np.array(mcl_L); mco = np.array(mco_L); dnm = (mcl - mco).tolist()
    coupling = float(np.mean(mcl - mco))

    # ---- U_info + projectors -----------------------------------------------------------------
    U = top_k_subspace(Hcl - Hco, K_UINFO)                   # [k, hidden]
    P_U = projector(U, ad.device)                            # [hidden, hidden]
    ident = torch.eye(P_U.shape[0], device=ad.device)
    rng = np.random.default_rng(SEED)
    P_rand = [projector(random_subspace(P_U.shape[0], K_UINFO, rng), ad.device) for _ in range(n_rand)]

    # ---- preflight memorials -----------------------------------------------------------------
    idem = float((P_U @ P_U - P_U).abs().max())
    orth = float((P_U @ (ident - P_U)).abs().max())
    decomp = float((dracc @ P_U + dracc @ (ident - P_U) - dracc).abs().max())
    checks = {
        "value_linearity": {"PASS": bool(lin_err < 1e-4), "max_abs_err": lin_err},
        "inject_identity": {"PASS": bool(inject_id_err < 1e-3), "max_abs_err": inject_id_err},
        "attention_row_fixed": {"PASS": True,
                                "note": "all payloads reconstruct from A+ (clean row); no patched "
                                        "forward is ever re-run, so the attention row is identical"},
        "projector_idempotent": {"PASS": bool(idem < 1e-5), "max_abs_err": idem},
        "projector_orthogonal": {"PASS": bool(orth < 1e-5), "max_abs_err": orth},
        "decomposition": {"PASS": bool(decomp < 1e-5), "max_abs_err": decomp},
        "probability_invariants": prob_inv,
        "positive_control_partner_coupling": {"PASS": bool(coupling >= 0.20), "coupling": coupling},
        "both_strands_present": {"PASS": len(strands) == 2, "strands": strands},
    }
    ckpt = Path(args.model); ckpt = ckpt if ckpt.is_absolute() else REPO / ckpt
    checks["checkpoint_recorded"] = {"PASS": True, "sha256": sha256_file(ckpt / "pytorch_model.bin")}
    pf = {"frozen": {"lin_tol": 1e-4, "proj_tol": 1e-5}, "commit": git_commit(REPO), "n": len(X),
          "K_uinfo": K_UINFO, "n_rand": n_rand, "checks": checks,
          "PASS": bool(all(c["PASS"] for c in checks.values()))}
    pf_path.parent.mkdir(parents=True, exist_ok=True)
    pf_path.write_text(json.dumps(pf, indent=1))
    print(f"preflight PASS={pf['PASS']}  -> {pf_path}")
    for name, c in pf["checks"].items():
        print(f"    {'ok ' if c['PASS'] else 'XX '}{name}"
              + (f"  ({c.get('max_abs_err')})" if "max_abs_err" in c else ""))

    if args.smoke:
        print("SMOKE: preflight/memorials only; no recovery magnitudes (val sealed).")
        return
    if not pf["PASS"] and not args.force:
        print("  *** preflight FAILED -- not reporting recoveries (prereg §3).")
        return

    # ---- payload delta-V builders (window value edits [B,K,Dh]) ------------------------------
    def dv(dr):
        return deliver(Wv, dr)

    def energy_match_scale(P_i, dr):
        eU = delivered_energy(dv(dr @ P_U))
        ei = delivered_energy(dv(dr @ P_i)).clamp_min(1e-12)
        return (eU / ei)[:, None, None]

    roll = torch.roll(dracc, shifts=len(X) // 2, dims=0)     # cross-gene mismatched payload

    # ---- injection loop (denoising: into the corrupt run) ------------------------------------
    num = {k: [] for k in ("address_floor", "full", "U", "perp", "nonsite", "crossgene")}
    num_rand = [[] for _ in range(n_rand)]
    noise_full, noise_perp = [], []                          # noising: into the clean run
    with torch.no_grad():
        for bi, i in enumerate(range(0, len(X), args.batch)):
            exs = X[i:i + args.batch]
            dn = [x.donor_pos for x in exs]
            sl = slice(i, i + len(exs))
            corr = corrupt_at(task, exs)
            clean = task.encode([x.sequence for x in exs], dn)
            bb, ba, bc = base[sl], Aacc[sl], Actl[sl]
            da, dc, dtk = dracc[sl], drctl[sl], dtok_all[sl]

            def inj(recon, enc):
                return inject_margin_donor(ad, task, enc, dn, dtk, recon)

            recon = {
                "address_floor": head_output(bb, ba, torch.zeros_like(dv(da))),
                "full": head_output(bb, ba, dv(da)),
                "U": head_output(bb, ba, dv(da @ P_U)),
                "perp": head_output(bb, ba, dv(da @ (ident - P_U))),
                "nonsite": head_output(bb, bc, dv(dc @ P_U)),
                "crossgene": head_output(bb, ba, dv(roll[sl] @ P_U)),
            }
            for k, r in recon.items():
                num[k] += (inj(r, corr) - mco[sl]).tolist()
            for j, P_i in enumerate(P_rand):
                s = energy_match_scale(P_i, da)
                r = head_output(bb, ba, s * dv(da @ P_i))
                num_rand[j] += (inj(r, corr) - mco[sl]).tolist()
            # noising into the CLEAN run: full (no-op memorial) and perp (U_info removed)
            noise_full += inj(recon["full"], clean).tolist()
            noise_perp += inj(recon["perp"], clean).tolist()
            print(f"  inject {min(i + args.batch, len(X))}/{len(X)}", end="\r", flush=True)
    print()

    # ---- statistics --------------------------------------------------------------------------
    def R(v):
        return ratio_of_means(v, dnm)

    def Rci(v):
        return {"recovery": R(v), "ci": list(gboot_ratio(v, dnm, genes))}

    R_rand = [R(v) for v in num_rand]
    R_U = R(num["U"])
    spec = R_U - float(np.mean(R_rand))
    perc = float(np.mean([R_U >= r for r in R_rand]))
    p_spec = empirical_upper_p_value(R_U, R_rand)
    N_U = R(num["full"]) - R(num["perp"])
    N_U_ci = list(gboot_ratio((np.array(num["full"]) - np.array(num["perp"])).tolist(), dnm, genes))
    # noising necessity: fraction of coupling lost when U_info removed from the clean value
    noise_loss = ratio_of_means((mcl - np.array(noise_perp)).tolist(), dnm)
    noise_fullkeep = ratio_of_means((mcl - np.array(noise_full)).tolist(), dnm)  # ~0 memorial

    result = {
        "n": len(X), "n_genes": len(set(genes)), "strands": strands, "coupling": coupling,
        "K_uinfo": K_UINFO, "n_rand": n_rand,
        "recoveries": {k: Rci(num[k]) for k in num},
        "note_full_scope": ("full restores CLEAN values at the ACCEPTOR WINDOW only (corrupt "
                            "elsewhere); it is the acceptor-window payload ceiling and is NOT A2's "
                            "joint (+0.37), which restored all positions. address_floor (base "
                            "A+.V-) cross-checks A2's address_only (~+0.01)."),
        "sufficiency_U": {"R_U": R_U, "E_random": float(np.mean(R_rand)),
                          "random_min": float(np.min(R_rand)), "random_max": float(np.max(R_rand)),
                          "specificity_U": spec, "U_percentile": perc, "add_one_p": p_spec},
        "necessity": {"N_U_denoise": N_U, "ci": N_U_ci, "R_perp": R(num["perp"]),
                      "N_U_noise_loss": noise_loss, "noise_full_keeps_clean": noise_fullkeep},
        "controls": {"nonsite": R(num["nonsite"]), "crossgene": R(num["crossgene"])},
        "preflight_PASS": pf["PASS"], "meets_power_target": bool(len(X) >= 250),
        "confirmatory": bool(pf["PASS"] and len(X) >= 250),
        "REFLECTION": {"status": "PENDING -- fill after inspection (prereg §5)"},
    }
    out.write_text(json.dumps(result, indent=1))

    status = "CONFIRMATORY" if result["confirmatory"] else "DIAGNOSTIC"
    man = ExperimentManifest.create(
        "splicebert_coupling:a2b_payload_localization", sys.argv, status=status,
        config={"split": args.split, "n": len(X), "n_genes": result["n_genes"], "K_uinfo": K_UINFO,
                "n_rand": n_rand, "reader": list(READER), "preflight_PASS": pf["PASS"],
                "meets_power_target": result["meets_power_target"]},
        seeds={"bootstrap": 0, "uinfo_svd": 0, "random_subspaces": SEED},
        notes=["Localizes L3H1's value payload to U_info vs an energy-matched random rank-8 "
               "distribution (post-LN value linearity -> analytic reconstruction; clean attention "
               "fixed). Behaviour-level; does not revisit R20.",
               f"status={status} (preflight_PASS={pf['PASS']}, power={result['meets_power_target']})."])
    man.add_input(str(ckpt / "pytorch_model.bin"))
    for data in (args.gtf, args.fasta):
        if Path(data).exists():
            man.add_input(data)
    man.write(out.parent / (out.stem + ".manifest.json"))

    print("=" * 80)
    for k in ("address_floor", "full", "U", "perp", "nonsite", "crossgene"):
        print(f"  {k:14s} R {result['recoveries'][k]['recovery']:+.3f}")
    s = result["sufficiency_U"]
    print(f"  U vs random: R_U {s['R_U']:+.3f}  E[rand] {s['E_random']:+.3f}  "
          f"specificity {s['specificity_U']:+.3f}  pct {s['U_percentile']:.2f}  p {s['add_one_p']:.3f}")
    print(f"  necessity N_U {N_U:+.3f} CI{N_U_ci}  R_perp {R(num['perp']):+.3f}  "
          f"noise_loss {noise_loss:+.3f} (full-keep {noise_fullkeep:+.3f})")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
