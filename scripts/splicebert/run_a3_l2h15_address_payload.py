#!/usr/bin/env python
"""Donor-graph experiment 7 -- A3: does L2H15 build L3H1's address or its payload?

Frozen protocol: studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_A3_L2H15_ADDRESS_PAYLOAD.md.

L2H15 supplies ~0.66 of the donor coupling, ~all through L3H1 (l2_accounting).  Does that flow
through L3H1's ADDRESS (attention row A) or its PAYLOAD (values V)?  Reconstruct L3H1's donor output
A.V with A/V drawn from L2H15-intact vs -ablated runs, inject it (donor only) into the L2H15-ablated
forward, and read the native coupling rescue.

Preflight/memorials FIRST; the run stops on failure.  --smoke = memorials only (val sealed).
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
from gfmcd.splicing.circuit import (  # noqa: E402
    ADDRESSER, CORRUPT_ALT, DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, READER, CouplingTask,
    ablate, capture_proj, gboot_ratio, load_adapter, load_examples, pick_device, ratio_of_means,
)
from gfmcd.splicing.coupling import substitute  # noqa: E402
from gfmcd.splicing.payload import inject_margin_donor  # noqa: E402

L3, H3 = READER
SPLIT_CHROMS = {"val": ("chr11", "chr16", "chr19"), "test": ("chr8", "chr9")}


def corrupt_at(task, exs):
    seqs = [substitute(x.sequence, x.next_acceptor_pos - 2, CORRUPT_ALT) for x in exs]
    return task.encode(seqs, [x.donor_pos for x in exs])


def value_head(ad, enc, Dh):
    v = capture_proj(ad, L3, ["value"], lambda: ad.model(**enc))["value"]
    return v[:, :, H3 * Dh:(H3 + 1) * Dh]                    # [B,S,Dh]


def recon(A_row, V):
    return torch.einsum("bs,bsd->bd", A_row, V)             # [B,Dh]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="results/splicebert_coupling/a3_l2h15_address_payload.json")
    args = ap.parse_args()

    out = REPO / args.out
    pf_path = out.parent / "preflight_a3_l2h15_address_payload.json"
    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=True)
    task = CouplingTask(ad, batch=args.batch)
    off = task.off
    Dh = ad.hidden_size // ad.num_heads

    n = min(args.n, 24) if args.smoke else args.n
    X, _ = load_examples(args.gtf, args.fasta, args.split, n, task=task, tag=f"a3-{args.split}",
                         min_effect=None)
    if len(X) < 8:
        raise RuntimeError(f"only {len(X)} eligible examples.")
    strands = sorted({x.strand for x in X})
    genes = [x.gene_id for x in X]
    print(f"{len(X)} eligible examples; strands={strands}; device={dev}"
          f"{'  [SMOKE]' if args.smoke else ''}")

    def margin(enc, dn):
        return task.margin(ad.model(**enc).logits.float(), dn).cpu().numpy()

    def att_row(enc, dtok):
        o = ad.model(**enc, output_attentions=True)
        b = torch.arange(o.attentions[L3].shape[0], device=ad.device)
        return o.attentions[L3][:, H3][b, dtok]

    CELLS = ["baseline", "address", "payload", "joint"]
    num = {c: [] for c in CELLS}
    dnm = []                                                # Y_intact - Y_ablated
    recon_err = 0.0; base_err = 0.0; prob_inv = None
    lift = []                                               # Y_intact - Y_ablated per example
    with torch.no_grad():
        for i in range(0, len(X), args.batch):
            exs = X[i:i + args.batch]
            dn = [x.donor_pos for x in exs]
            dtok = torch.tensor([d + off for d in dn], device=ad.device)
            clean = task.encode([x.sequence for x in exs], dn)
            corr = corrupt_at(task, exs)

            # intact captures
            Ap_cl, Ap_co = att_row(clean, dtok), att_row(corr, dtok)
            Vp_cl, Vp_co = value_head(ad, clean, Dh), value_head(ad, corr, Dh)
            m_int_cl, m_int_co = margin(clean, dn), margin(corr, dn)
            Y_int = m_int_cl - m_int_co

            with ablate(ad, [ADDRESSER])():                 # L2H15 mean-ablated for all below
                Am_cl, Am_co = att_row(clean, dtok), att_row(corr, dtok)
                Vm_cl, Vm_co = value_head(ad, clean, Dh), value_head(ad, corr, Dh)
                m_abl_cl, m_abl_co = margin(clean, dn), margin(corr, dn)
                Y_abl = m_abl_cl - m_abl_co

                combos = {
                    "baseline": (Am_cl, Vm_cl, Am_co, Vm_co),
                    "address": (Ap_cl, Vm_cl, Ap_co, Vm_co),
                    "payload": (Am_cl, Vp_cl, Am_co, Vp_co),
                    "joint": (Ap_cl, Vp_cl, Ap_co, Vp_co),
                }
                for name, (acl, vcl, aco, vco) in combos.items():
                    m_cl = inject_margin_donor(ad, task, clean, dn, dtok, recon(acl, vcl))
                    m_co = inject_margin_donor(ad, task, corr, dn, dtok, recon(aco, vco))
                    num[name] += ((m_cl - m_co) - Y_abl).tolist()

                if prob_inv is None:                        # memorials on the first batch
                    b = torch.arange(len(exs), device=ad.device)
                    head_abl = capture_head_outputs(ad, lambda: ad.model(**clean))[L3][b, H3, dtok]
                    recon_err = float((recon(Am_cl, Vm_cl) - head_abl).abs().max())
                    base_err = float(np.abs(inject_margin_donor(
                        ad, task, clean, dn, dtok, recon(Am_cl, Vm_cl)) - m_abl_cl).max())
                    rs = float((Ap_cl.sum(-1) - 1).abs().max())
                    rs_a = float((Am_cl.sum(-1) - 1).abs().max())
                    prob_inv = {"PASS": bool(max(rs, rs_a) < 1e-3 and float(Ap_cl.max()) > 0),
                                "max_row_sum_err": max(rs, rs_a)}
            dnm += (Y_int - Y_abl).tolist()
            lift += (Y_int - Y_abl).tolist()
            print(f"  {min(i + args.batch, len(X))}/{len(X)}", end="\r", flush=True)
    print()

    lift_mean = float(np.mean(lift))
    checks = {
        "reconstruction_matches": {"PASS": bool(recon_err < 1e-4), "max_abs_err": recon_err},
        "baseline_reproduces_ablated": {"PASS": bool(base_err < 1e-3), "max_abs_err": base_err},
        "probability_invariants": prob_inv,
        "positive_control_L2H15_contributes": {"PASS": bool(lift_mean >= 0.20),
                                               "Y_intact_minus_ablated": lift_mean},
        "joint_rescues": {"PASS": bool(ratio_of_means(num["joint"], dnm) >= 0.30),
                          "R_joint": ratio_of_means(num["joint"], dnm)},
        "both_strands_present": {"PASS": len(strands) == 2, "strands": strands},
    }
    ckpt = Path(args.model); ckpt = ckpt if ckpt.is_absolute() else REPO / ckpt
    checks["checkpoint_recorded"] = {"PASS": True, "sha256": sha256_file(ckpt / "pytorch_model.bin")}
    pf = {"commit": git_commit(REPO), "n": len(X), "checks": checks,
          "PASS": bool(all(c["PASS"] for c in checks.values()))}
    pf_path.parent.mkdir(parents=True, exist_ok=True)
    pf_path.write_text(json.dumps(pf, indent=1))
    print(f"preflight PASS={pf['PASS']}  -> {pf_path}")
    for name, c in pf["checks"].items():
        print(f"    {'ok ' if c['PASS'] else 'XX '}{name}"
              + (f"  ({c.get('max_abs_err')})" if "max_abs_err" in c else ""))

    if args.smoke:
        print("SMOKE: preflight/memorials only; no rescue magnitudes (val sealed).")
        return
    if not pf["PASS"] and not args.force:
        print("  *** preflight FAILED -- not reporting rescues (prereg §3).")
        return

    def Rci(name):
        return {"rescue": ratio_of_means(num[name], dnm),
                "ci": list(gboot_ratio(num[name], dnm, genes))}
    R = {c: Rci(c) for c in CELLS}
    a = {c: np.array(num[c]) for c in CELLS}
    inter = ratio_of_means((a["joint"] - a["address"] - a["payload"] + a["baseline"]).tolist(), dnm)
    result = {
        "n": len(X), "n_genes": len(set(genes)), "strands": strands,
        "Y_intact_minus_ablated": lift_mean, "rescues": R, "interaction_AxV": inter,
        "preflight_PASS": pf["PASS"], "meets_power_target": bool(len(X) >= 250),
        "confirmatory": bool(pf["PASS"] and len(X) >= 250),
        "REFLECTION": {"status": "PENDING -- fill after inspection (prereg §5)"},
    }
    out.write_text(json.dumps(result, indent=1))
    status = "CONFIRMATORY" if result["confirmatory"] else "DIAGNOSTIC"
    man = ExperimentManifest.create(
        "splicebert_coupling:a3_l2h15_address_payload", sys.argv, status=status,
        config={"split": args.split, "n": len(X), "n_genes": result["n_genes"],
                "reader": list(READER), "addresser": list(ADDRESSER), "preflight_PASS": pf["PASS"],
                "meets_power_target": result["meets_power_target"]},
        seeds={"bootstrap": 0},
        notes=["L2H15 address-vs-payload attribution: reconstruct-inject L3H1's donor output with "
               "A/V from L2H15-intact vs -ablated, read coupling rescue. Behaviour-level.",
               f"status={status} (preflight_PASS={pf['PASS']}, power={result['meets_power_target']})."])
    man.add_input(str(ckpt / "pytorch_model.bin"))
    for data in (args.gtf, args.fasta):
        if Path(data).exists():
            man.add_input(data)
    man.write(out.parent / (out.stem + ".manifest.json"))

    print("=" * 80)
    for c in CELLS:
        print(f"  {c:10s} rescue {R[c]['rescue']:+.3f}  CI[{R[c]['ci'][0]:+.2f},{R[c]['ci'][1]:+.2f}]")
    print(f"  interaction A×V {inter:+.3f}   (L2H15 lift {lift_mean:+.3f})")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
