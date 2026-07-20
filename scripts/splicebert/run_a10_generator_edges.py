#!/usr/bin/env python
"""Donor-graph experiment 14 -- A10: head-level EDGES of the acceptor generator.

Frozen protocol: studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_A10_GENERATOR_EDGES.md.

Reuses the A8 value-path metric (address held clean; reconstruct-inject) EXACTLY, with edge-resolving
conditions: held-out L0-head resolution + top-head interactions; head->MLP0 mediation; MLP serial chain.
No new kernel.  --smoke = memorials only (val sealed).
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
sys.path.insert(0, str(REPO / "scripts" / "splicebert"))

from run_a7_source_localization import capture as a7cap, clamp_write_mask     # noqa: E402
from gfmcd.provenance import ExperimentManifest, git_commit, sha256_file       # noqa: E402
from gfmcd.circuits.hf_bert import patch_heads                                 # noqa: E402
from gfmcd.splicing.circuit import (                                          # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, gboot_ratio,
    load_adapter, load_examples, pick_device,
)
from gfmcd.splicing.payload import deliver, head_output, inject_margin_donor, value_weight_head  # noqa: E402

L3, H3 = 3, 1
C_HEADS = [(1, 12), (2, 15), (3, 1), (4, 0), (4, 2), (5, 11)]
ACC_OFF = (-3, -2, -1, 0)
L0_TOP = [7, 12, 10]     # A8's strongest L0 heads (for the interaction conditions)
SEED = 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=220)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF); ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="results/splicebert_coupling/a10_generator_edges.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    out = REPO / args.out
    pf_path = out.parent / "preflight_a10_generator_edges.json"
    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=True)
    task = CouplingTask(ad, batch=args.batch)
    off = task.off; Dh = ad.hidden_size // ad.num_heads
    Wv = value_weight_head(ad).to(dev)

    n = min(args.n, 24) if args.smoke else args.n
    X, _ = load_examples(args.gtf, args.fasta, args.split, n, task=task, tag=f"a10-{args.split}",
                         min_effect=None)
    if len(X) < 16:
        raise RuntimeError(f"only {len(X)} eligible examples.")
    strands = sorted({x.strand for x in X})

    A_HEADS = [(l, h) for l in (0, 1, 2) for h in range(ad.num_heads) if (l, h) not in C_HEADS]
    A_MLPS = [(0, "mlp"), (1, "mlp")]
    A_NODES = A_HEADS + A_MLPS
    L0H = [(0, h) for h in range(ad.num_heads)]
    ALL_HEADS = [(l, h) for l in range(ad.num_layers) for h in range(ad.num_heads)]
    ALL_MLPS = [(l, "mlp") for l in range(ad.num_layers)]

    conds = {"empty": ([], None), "full": (A_NODES, None)}
    if not args.smoke:
        for h in range(ad.num_heads):
            conds[f"no_L0_{h}"] = ([a for a in A_NODES if a != (0, h)], None)
        conds["no_MLP0"] = ([a for a in A_NODES if a != (0, "mlp")], None)
        conds["no_MLP1"] = ([a for a in A_NODES if a != (1, "mlp")], None)
        conds["circ_MLP2"] = (A_NODES, [((2, "mlp"), "ACC")])
        conds["L0heads_only"] = (L0H, None)
        conds["L0heads_MLP0"] = (L0H + [(0, "mlp")], None)
        conds["MLP0_only"] = ([(0, "mlp")], None)
        conds["MLP1_only"] = ([(1, "mlp")], None)
        conds["MLP0_MLP1"] = ([(0, "mlp"), (1, "mlp")], None)
        for a, b in [(7, 12), (7, 10), (12, 10)]:
            conds[f"no2_{a}_{b}"] = ([x for x in A_NODES if x not in [(0, a), (0, b)]], None)

    Mp = {k: [] for k in conds}
    lin_err = 0.0; mem = {"empty_is_corrupt": 0.0, "full_is_clean": 0.0}
    mcl_all, mco_all, genes = [], [], []

    for i in range(0, len(X), args.batch):
        ch = X[i:i + args.batch]
        cl, co, dn = task.pair(ch)
        B, S = cl["input_ids"].shape
        dtok = torch.tensor([d + off for d in dn], device=dev)
        accwin = torch.tensor([[x.next_acceptor_pos + off + k for k in ACC_OFF] for x in ch], device=dev)
        bidx = torch.arange(B, device=dev)[:, None].expand(B, 4)
        ACCm = torch.zeros(B, S, dtype=torch.bool, device=dev); ACCm[bidx, accwin] = True
        notACC, allT = ~ACCm, torch.ones(B, S, dtype=torch.bool, device=dev)

        def caprun(inp, want_attn):
            vbox = {}
            hh = ad.attention_self_modules[L3].value.register_forward_hook(
                lambda _m, _i, o: vbox.__setitem__("v", o.detach()))
            try:
                o = ad.model(**inp, output_hidden_states=True, output_attentions=want_attn)
            finally:
                hh.remove()
            vhead = vbox["v"][:, :, H3 * Dh:(H3 + 1) * Dh]
            ap = o.attentions[L3][torch.arange(B, device=dev), H3, dtok, :] if want_attn else None
            return o, o.hidden_states[L3], vhead, ap

        o_cl, r3_cl, V_cl, Aplus = caprun(cl, True)
        o_co, r3_co, V_co, _ = caprun(co, False)
        m_cl = task.margin(o_cl.logits.float(), dn); m_co = task.margin(o_co.logits.float(), dn)
        base = torch.einsum("bs,bsd->bd", Aplus, V_co)
        Aplus_acc = Aplus[bidx, accwin]; r3co_acc = r3_co[bidx, accwin]
        lin_err = max(lin_err, float(((V_cl - V_co)[bidx, accwin] - deliver(Wv, (r3_cl - r3_co)[bidx, accwin])).abs().max()))

        _, hco, mco_w, eco = a7cap(ad, co)
        _, hcl, mcl_w, ecl = a7cap(ad, cl)

        def scrub_r3(open_nodes, extra):
            closed = [a for a in A_NODES if a not in open_nodes]
            ho = [x for x in open_nodes if isinstance(x[1], int)]; mo = [x for x in open_nodes if x[1] == "mlp"]
            hc = [x for x in closed if isinstance(x[1], int)]; mc = [x for x in closed if x[1] == "mlp"]
            with ExitStack() as st:
                if ho: st.enter_context(patch_heads(ad, hco, ho, token_mask=notACC))
                if hc: st.enter_context(patch_heads(ad, hco, hc, token_mask=allT))
                if mo: st.enter_context(clamp_write_mask(ad, mo, mco_w, notACC))
                if mc: st.enter_context(clamp_write_mask(ad, mc, mco_w, allT))
                for node, _m in (extra or []):
                    if isinstance(node[1], int):
                        st.enter_context(patch_heads(ad, hco, [node], token_mask=ACCm))
                    else:
                        st.enter_context(clamp_write_mask(ad, [node], mco_w, ACCm))
                return ad.model(**cl, output_hidden_states=True).hidden_states[L3]

        def mprime(r3S):
            od = head_output(base, Aplus_acc, deliver(Wv, r3S[bidx, accwin] - r3co_acc))
            return inject_margin_donor(ad, task, co, dn, dtok, od)

        # scrub-cover memorials
        def full_scrub(hs, ms, es, binp):
            with ExitStack() as st:
                st.enter_context(patch_heads(ad, hs, ALL_HEADS, token_mask=allT))
                st.enter_context(clamp_write_mask(ad, ALL_MLPS, ms, allT))
                from gfmcd.splicing.provenance import clamp_embed
                st.enter_context(clamp_embed(ad, es, torch.arange(S, device=dev)[None].expand(B, S)))
                return task.margin(ad.model(**binp).logits.float(), dn)
        me = full_scrub(hco, mco_w, eco, cl); mf = full_scrub(hcl, mcl_w, ecl, co)
        mem["empty_is_corrupt"] = max(mem["empty_is_corrupt"], float((me - m_co).abs().max()))
        mem["full_is_clean"] = max(mem["full_is_clean"], float((mf - m_cl).abs().max()))

        for name, (open_nodes, extra) in conds.items():
            Mp[name].extend(mprime(scrub_r3(open_nodes, extra)).tolist())
        mcl_all.extend(m_cl.cpu().tolist()); mco_all.extend(m_co.cpu().tolist())
        genes.extend([x.gene_id for x in ch])
        print(f"  batch {min(i + args.batch, len(X))}/{len(X)}", end="\r", flush=True)
    print()

    mco = np.array(mco_all)
    M0 = float(np.mean(Mp["empty"]) - mco.mean()); MA = float(np.mean(Mp["full"]) - mco.mean())
    coupling = float(np.mean(np.array(mcl_all) - mco))
    ug = sorted(set(genes)); Dg = set(ug[::2])
    inD = np.array([g in Dg for g in genes]); inH = ~inD

    checks = {
        "value_linearity": {"PASS": bool(lin_err < 1e-4), "max_abs_err": lin_err},
        "empty_circuit_is_corrupt": {"PASS": bool(mem["empty_is_corrupt"] < 1e-4), "max_abs_err": mem["empty_is_corrupt"]},
        "full_circuit_is_clean": {"PASS": bool(mem["full_is_clean"] < 1e-4), "max_abs_err": mem["full_is_clean"]},
        "generator_ceiling": {"PASS": bool(MA - M0 >= 0.10), "M_empty": M0, "M_full": MA, "ceiling": MA - M0},
        "positive_control_partner_coupling": {"PASS": bool(coupling >= 0.20), "coupling": coupling},
        "both_genes_split": {"PASS": bool(inD.sum() >= (5 if args.smoke else 40) and inH.sum() >= (5 if args.smoke else 40)),
                             "nD": int(inD.sum()), "nH": int(inH.sum())},
        "both_strands_present": {"PASS": len(strands) == 2, "strands": strands},
    }
    ckpt = Path(args.model); ckpt = ckpt if ckpt.is_absolute() else REPO / ckpt
    checks["checkpoint_recorded"] = {"PASS": True, "sha256": sha256_file(ckpt / "pytorch_model.bin")}
    pf = {"commit": git_commit(REPO), "n": len(X), "checks": checks}
    pf["PASS"] = bool(all(c["PASS"] for c in pf["checks"].values()))
    pf_path.parent.mkdir(parents=True, exist_ok=True); pf_path.write_text(json.dumps(pf, indent=1))
    print(f"preflight PASS={pf['PASS']}  -> {pf_path}")
    for name, c in pf["checks"].items():
        print(f"    {'ok ' if c['PASS'] else 'XX '}{name}" + (f"  ({c.get('max_abs_err', c.get('ceiling',''))})" if ('max_abs_err' in c or 'ceiling' in c) else ""))

    if args.smoke:
        print("SMOKE: memorials only (val sealed)."); return
    if not pf["PASS"] and not args.force:
        print("  *** preflight FAILED -- not reporting (prereg s3)."); return

    emp = np.array(Mp["empty"]); ful = np.array(Mp["full"])
    def R(name, mask):
        return float((np.array(Mp[name])[mask].mean() - emp[mask].mean()) / (ful[mask].mean() - emp[mask].mean()))
    def nec(drop, mask):
        g = [gg for gg, m in zip(genes, mask) if m]
        num = (ful - np.array(Mp[drop]))[mask].tolist(); den = (ful - emp)[mask].tolist()
        return float(np.mean(num) / np.mean(den)), list(gboot_ratio(num, den, g))

    # L0-head resolution (D and H)
    l0 = {}
    for h in range(ad.num_heads):
        NhD, ciD = nec(f"no_L0_{h}", inD); NhH, ciH = nec(f"no_L0_{h}", inH)
        l0[f"L0H{h}"] = {"N_h_D": NhD, "ci_D": ciD, "N_h_H": NhH, "ci_H": ciH,
                         "load_bearing_H": bool(ciH[0] > 0)}
    L0star = [k for k, v in l0.items() if v["load_bearing_H"]]

    # interactions among the top L0 heads
    inter = {}
    for a, b in [(7, 12), (7, 10), (12, 10)]:
        I = R("full", inD) - R(f"no_L0_{a}", inD) - R(f"no_L0_{b}", inD) + R(f"no2_{a}_{b}", inD)
        inter[f"L0H{a}xL0H{b}"] = I

    mediation = {"R_L0heads_only_H": R("L0heads_only", inH), "R_L0heads_MLP0_H": R("L0heads_MLP0", inH)}
    mediation["mediated_by_MLP0"] = bool(mediation["R_L0heads_only_H"] < 0.5 * max(mediation["R_L0heads_MLP0_H"], 1e-9))
    mlpchain = {"R_MLP0_only_H": R("MLP0_only", inH), "R_MLP1_only_H": R("MLP1_only", inH),
                "R_MLP0_MLP1_H": R("MLP0_MLP1", inH),
                "N_MLP0_H": nec("no_MLP0", inH), "N_MLP1_H": nec("no_MLP1", inH),
                "N_MLP2_H": nec("circ_MLP2", inH)}
    serial_chain = bool(mlpchain["R_MLP1_only_H"] < 0.10 and mlpchain["N_MLP0_H"][1][0] > 0
                        and mlpchain["N_MLP1_H"][1][0] > 0 and mlpchain["N_MLP2_H"][1][0] > 0)

    result = {
        "n": len(X), "n_genes": len(ug), "nD": int(inD.sum()), "nH": int(inH.sum()), "coupling": coupling,
        "M_empty_rec": M0, "M_full_rec": MA, "ceiling": MA - M0,
        "R_full_H": R("full", inH), "l0_heads": l0, "L0_star_heldout": L0star,
        "top_head_interactions_D": inter, "mediation": mediation, "mlp_chain": mlpchain,
        "serial_chain_confirmed": serial_chain,
        "preflight_PASS": pf["PASS"], "meets_power_target": bool(len(X) >= 150 and inD.sum() >= 40 and inH.sum() >= 40),
        "confirmatory": bool(pf["PASS"] and len(X) >= 150),
        "REFLECTION": {"status": "PENDING -- fill after inspection (prereg s5)"},
    }
    out.write_text(json.dumps(result, indent=1))
    status = "CONFIRMATORY" if result["confirmatory"] else "DIAGNOSTIC"
    man = ExperimentManifest.create(
        "splicebert_coupling:a10_generator_edges", sys.argv, status=status,
        config={"split": args.split, "n": len(X), "n_genes": len(ug), "preflight_PASS": pf["PASS"],
                "meets_power_target": result["meets_power_target"]},
        seeds={"gene_split": SEED, "gene_bootstrap": 0},
        notes=["Head-level edges of the acceptor generator (A8 value-path metric). Held-out L0-head "
               "resolution + interactions; head->MLP0 mediation; MLP serial chain. No new kernel.",
               f"status={status}."])
    man.add_input(str(ckpt / "pytorch_model.bin")); man.write(out.parent / (out.stem + ".manifest.json"))

    print("=" * 84)
    print(f"  ceiling={MA - M0:+.3f}  R_full_H={R('full', inH):+.3f}")
    top = sorted(l0.items(), key=lambda kv: -kv[1]["N_h_H"])[:6]
    print("  L0 heads (held-out N_h): " + "  ".join(f"{k}={v['N_h_H']:+.3f}{'*' if v['load_bearing_H'] else ''}" for k, v in top))
    print(f"  L0* (held-out load-bearing): {L0star}")
    print(f"  interactions(D): " + "  ".join(f"{k}={v:+.3f}" for k, v in inter.items()))
    print(f"  mediation: L0heads_only={mediation['R_L0heads_only_H']:+.3f} +MLP0={mediation['R_L0heads_MLP0_H']:+.3f} mediated={mediation['mediated_by_MLP0']}")
    print(f"  MLP chain: MLP0_only={mlpchain['R_MLP0_only_H']:+.3f} MLP1_only={mlpchain['R_MLP1_only_H']:+.3f} MLP0+1={mlpchain['R_MLP0_MLP1_H']:+.3f}; serial_confirmed={serial_chain}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
